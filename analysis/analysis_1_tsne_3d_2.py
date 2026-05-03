"""Analysis 1 (3D version) — sharp-green success + blue->red gradient for failure.

Key change vs paper-style: success rollouts are colored sharp GREEN (not blue).
Failure rollouts use blue (early timesteps) -> red (late timesteps) gradient.
This makes the 3 "kinds" of points visually distinct:
  - Green = success
  - Blue  = early-timestep failure
  - Red   = late-timestep failure
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    add_shared_args,
    load_all_rollouts,
    filter_by_tasks,
    make_output_dir,
    print_data_summary,
    aggregate_tokens,
)

# ---- COLOR CONSTANTS ----
SUCCESS_GREEN = "#2ca02c"    # sharp tab:green for successful rollouts
FAILURE_CMAP = "coolwarm"    # blue -> white -> red gradient for failures


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 1 (3D, green-success): 3D viz of VLA features.")
    add_shared_args(p)
    p.add_argument("--method", choices=["tsne", "pca", "umap"], default="tsne")
    p.add_argument("--pca_first", action="store_true", default=False,
                   help="Pre-reduce to 50 dims with PCA before t-SNE. Default OFF (paper-matching).")
    p.add_argument("--feat_skip", type=int, default=1,
                   help="Take every N-th timestep. Default 1 = keep ALL.")
    p.add_argument("--color_mode", choices=["binary", "temporal_gradient"],
                   default="temporal_gradient",
                   help="binary: green=success, red=failure. "
                        "temporal_gradient: green=success, blue->red gradient over time for failures. "
                        "(Falls back to binary in aggregate modes.)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_html", action="store_true",
                   help="Skip the interactive Plotly HTML output (PNG only).")
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-60.0)
    return p.parse_args()


def reduce_to_3d(X: np.ndarray, args) -> np.ndarray:
    if args.method == "pca":
        print("Running PCA -> 3D ...")
        return PCA(n_components=3, random_state=args.seed).fit_transform(X)

    if args.method == "umap":
        import umap
        print(f"Running UMAP on {X.shape} -> 3D ...")
        return umap.UMAP(n_components=3, random_state=args.seed).fit_transform(X)

    if args.pca_first and X.shape[1] > 50:
        print(f"Optional PCA pre-reduction: {X.shape[1]} -> 50 dims")
        X = PCA(n_components=50, random_state=args.seed).fit_transform(X)

    print(f"Running 3D t-SNE on {X.shape} (perplexity={args.perplexity}). May take a while ...")
    tsne = TSNE(n_components=3, perplexity=args.perplexity,
                random_state=args.seed, verbose=1)
    return tsne.fit_transform(X)


def build_paper_style_points(rollouts, token_agg, feat_skip, timestep_mode, start_step):
    feats_list, labels_list, tids_list, rids_list, ts_list = [], [], [], [], []

    for ri, r in enumerate(rollouts):
        hs = aggregate_tokens(r["hidden_states"], token_agg)
        T = hs.shape[0]

        if timestep_mode == "per_step":
            keep_ts = list(range(0, T, feat_skip))
        elif timestep_mode == "aggregate":
            feats_list.append(hs.mean(axis=0)[None, :])
            labels_list.append([r["label"]]); tids_list.append([r["task_id"]])
            rids_list.append([ri]); ts_list.append([-1])
            continue
        elif timestep_mode == "late_per_step":
            keep_ts = list(range(start_step, T, feat_skip))
        elif timestep_mode == "late_aggregate":
            feats_list.append(hs[start_step:].mean(axis=0)[None, :])
            labels_list.append([r["label"]]); tids_list.append([r["task_id"]])
            rids_list.append([ri]); ts_list.append([-1])
            continue
        else:
            raise ValueError(f"Unknown timestep_mode: {timestep_mode}")

        if len(keep_ts) == 0:
            continue
        sub = hs[keep_ts]
        feats_list.append(sub)
        labels_list.append([r["label"]] * len(keep_ts))
        tids_list.append([r["task_id"]] * len(keep_ts))
        rids_list.append([ri] * len(keep_ts))
        ts_list.append(keep_ts)

    feats = np.concatenate(feats_list, axis=0).astype(np.float32)
    labels = np.array([x for sub in labels_list for x in sub], dtype=np.int64)
    task_ids = np.array([x for sub in tids_list for x in sub], dtype=np.int64)
    rollout_ids = np.array([x for sub in rids_list for x in sub], dtype=np.int64)
    timesteps = np.array([x for sub in ts_list for x in sub], dtype=np.int64)
    return feats, labels, task_ids, rollout_ids, timesteps


def compute_failure_progression(rollouts, rollout_ids):
    """For failures only, return progression in [0,1] over kept timesteps. NaN for successes."""
    progression = np.full(len(rollout_ids), np.nan, dtype=np.float32)
    for ri, r in enumerate(rollouts):
        mask = rollout_ids == ri
        n = int(mask.sum())
        if n == 0 or r["label"] == 0:
            continue
        if n == 1:
            progression[mask] = 1.0
        else:
            progression[mask] = np.linspace(0, 1, n, dtype=np.float32)
    return progression


# ============================================================================
# Static PNG plots (Matplotlib 3D)
# ============================================================================

def plot_3d_binary(X3d, labels, save_path, title, elev=20, azim=-60):
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    succ_mask = labels == 0
    fail_mask = labels == 1
    ax.scatter(X3d[succ_mask, 0], X3d[succ_mask, 1], X3d[succ_mask, 2],
               c=SUCCESS_GREEN, label=f"Success (n={succ_mask.sum()})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.scatter(X3d[fail_mask, 0], X3d[fail_mask, 1], X3d[fail_mask, 2],
               c="#d62728", label=f"Failure (n={fail_mask.sum()})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(markerscale=2.5, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_3d_temporal_with_green_success(X3d, labels, progression, save_path, title, elev=20, azim=-60):
    """Two scatter calls:
       1) Successes in sharp green
       2) Failures with blue->red colormap
    """
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    succ_mask = labels == 0
    fail_mask = labels == 1

    # Success points: sharp green
    ax.scatter(
        X3d[succ_mask, 0], X3d[succ_mask, 1], X3d[succ_mask, 2],
        c=SUCCESS_GREEN, label=f"Success (n={succ_mask.sum()})",
        s=8, alpha=0.6, edgecolors="none", depthshade=True,
    )

    # Failure points: temporal gradient
    sc = ax.scatter(
        X3d[fail_mask, 0], X3d[fail_mask, 1], X3d[fail_mask, 2],
        c=progression[fail_mask], cmap=FAILURE_CMAP,
        vmin=0.0, vmax=1.0,
        s=8, alpha=0.6, edgecolors="none", depthshade=True,
        label=f"Failure (n={fail_mask.sum()})",
    )

    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title + "\nGreen=success | Blue=early-failure | Red=late-failure")
    ax.view_init(elev=elev, azim=azim)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Failure progression (0=early -> 1=late)")
    ax.legend(markerscale=2.5, fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_3d_by_task(X3d, task_ids, save_path, title, elev=20, azim=-60):
    unique_tasks = sorted(set(task_ids.tolist()))
    cmap = mpl.cm.get_cmap("tab10", len(unique_tasks))
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        ax.scatter(X3d[mask, 0], X3d[mask, 1], X3d[mask, 2],
                   c=[cmap(i)], label=f"Task {tid}",
                   s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(markerscale=2.5, loc="best", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Interactive HTML plots (Plotly)
# ============================================================================

def plot_3d_interactive_binary(X3d, labels, rollouts, rollout_ids, save_path, title):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML. Run: pip install plotly")
        return
    succ_mask = labels == 0
    fail_mask = labels == 1
    hover_texts = [
        f"Task {rollouts[ri]['task_id']}: {rollouts[ri]['task_description']}"
        f"<br>Episode {rollouts[ri]['episode_idx']}"
        f"<br>{'Success' if rollouts[ri]['label']==0 else 'Failure'}"
        for ri in rollout_ids
    ]
    hover_succ = [hover_texts[i] for i in np.where(succ_mask)[0]]
    hover_fail = [hover_texts[i] for i in np.where(fail_mask)[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X3d[succ_mask, 0], y=X3d[succ_mask, 1], z=X3d[succ_mask, 2],
        mode="markers",
        marker=dict(size=3, color=SUCCESS_GREEN, opacity=0.65),
        name=f"Success (n={succ_mask.sum()})",
        text=hover_succ, hoverinfo="text",
    ))
    fig.add_trace(go.Scatter3d(
        x=X3d[fail_mask, 0], y=X3d[fail_mask, 1], z=X3d[fail_mask, 2],
        mode="markers",
        marker=dict(size=3, color="#d62728", opacity=0.65),
        name=f"Failure (n={fail_mask.sum()})",
        text=hover_fail, hoverinfo="text",
    ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        width=1100, height=900,
    )
    fig.write_html(str(save_path))
    print(f"Saved interactive HTML: {save_path}")


def plot_3d_interactive_temporal_with_green_success(
    X3d, labels, progression, rollouts, rollout_ids, timesteps, save_path, title
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML.")
        return

    succ_mask = labels == 0
    fail_mask = labels == 1

    def make_hover(idxs):
        return [
            f"Task {rollouts[rollout_ids[j]]['task_id']}: "
            f"{rollouts[rollout_ids[j]]['task_description']}"
            f"<br>Episode {rollouts[rollout_ids[j]]['episode_idx']}"
            f"<br>{'Success' if rollouts[rollout_ids[j]]['label']==0 else 'Failure'}"
            f"<br>Timestep: {timesteps[j]}"
            for j in idxs
        ]

    fig = go.Figure()

    # Success trace: solid green
    succ_idxs = np.where(succ_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=X3d[succ_mask, 0], y=X3d[succ_mask, 1], z=X3d[succ_mask, 2],
        mode="markers",
        marker=dict(size=3, color=SUCCESS_GREEN, opacity=0.65),
        name=f"Success (n={succ_mask.sum()})",
        text=make_hover(succ_idxs), hoverinfo="text",
    ))

    # Failure trace: blue->red gradient by progression
    fail_idxs = np.where(fail_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=X3d[fail_mask, 0], y=X3d[fail_mask, 1], z=X3d[fail_mask, 2],
        mode="markers",
        marker=dict(
            size=3, opacity=0.65,
            color=progression[fail_mask], colorscale="RdBu_r",
            cmin=0.0, cmax=1.0,
            colorbar=dict(title="Failure progression<br>(0=early, 1=late)"),
        ),
        name=f"Failure (n={fail_mask.sum()})",
        text=make_hover(fail_idxs), hoverinfo="text",
    ))

    fig.update_layout(
        title=title + "<br>Green=success | Blue=early-failure | Red=late-failure",
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        width=1100, height=900,
    )
    fig.write_html(str(save_path))
    print(f"Saved interactive HTML: {save_path}")


def plot_3d_interactive_by_task(X3d, task_ids, rollouts, rollout_ids, save_path, title):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML.")
        return
    unique_tasks = sorted(set(task_ids.tolist()))
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
               "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    fig = go.Figure()
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        idxs = np.where(mask)[0]
        hover_texts = [
            f"Task {tid}: {rollouts[rollout_ids[j]]['task_description']}"
            f"<br>Episode {rollouts[rollout_ids[j]]['episode_idx']}"
            f"<br>{'Success' if rollouts[rollout_ids[j]]['label']==0 else 'Failure'}"
            for j in idxs
        ]
        fig.add_trace(go.Scatter3d(
            x=X3d[mask, 0], y=X3d[mask, 1], z=X3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=palette[i % len(palette)], opacity=0.65),
            name=f"Task {tid}",
            text=hover_texts, hoverinfo="text",
        ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        width=1100, height=900,
    )
    fig.write_html(str(save_path))
    print(f"Saved interactive HTML: {save_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    if args.output_dir is None:
        out_dir = make_output_dir("analysis_1_tsne_3d_green")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_all_rollouts(args.rollout_root)
    rollouts = filter_by_tasks(rollouts, args.task_filter)
    rollouts = sorted(rollouts, key=lambda r: (r["task_id"], r["episode_idx"]))

    feats, labels, task_ids, rollout_ids, timesteps = build_paper_style_points(
        rollouts, args.token_agg, args.feat_skip, args.timestep_mode, args.start_step
    )

    fake_args = argparse.Namespace(
        token_agg=args.token_agg,
        timestep_mode=args.timestep_mode,
        start_step=args.start_step,
    )
    print_data_summary(rollouts, feats, labels, task_ids, fake_args)

    effective_color_mode = args.color_mode
    if args.color_mode == "temporal_gradient" and args.timestep_mode in ("aggregate", "late_aggregate"):
        print(f"  NOTE: timestep_mode={args.timestep_mode} has 1 point per rollout.")
        print(f"        Falling back to binary coloring.\n")
        effective_color_mode = "binary"

    print(f"  feat_skip:   {args.feat_skip}")
    print(f"  pca_first:   {args.pca_first}")
    print(f"  color_mode:  {args.color_mode}{' -> binary (auto)' if effective_color_mode != args.color_mode else ''}")
    print(f"  Reducing to 3D...\n")

    X3d = reduce_to_3d(feats, args)

    np.savez(
        out_dir / "embedding_3d.npz",
        X3d=X3d, labels=labels, task_ids=task_ids,
        rollout_ids=rollout_ids, timesteps=timesteps,
        args=str(vars(args)),
    )

    base_title = (
        f"{args.method.upper()} 3D | token={args.token_agg} | "
        f"mode={args.timestep_mode} | feat_skip={args.feat_skip}"
    )
    if "late" in args.timestep_mode:
        base_title += f" (start={args.start_step})"

    # Static PNG: by task
    plot_3d_by_task(X3d, task_ids, out_dir / "plot_by_task.png",
                    f"{base_title}\nColored by Task ID",
                    elev=args.elev, azim=args.azim)

    # Static PNG: by label
    if effective_color_mode == "temporal_gradient":
        progression = compute_failure_progression(rollouts, rollout_ids)
        plot_3d_temporal_with_green_success(
            X3d, labels, progression, out_dir / "plot_by_label.png",
            f"{base_title}\n(Green-success + temporal-gradient failure)",
            elev=args.elev, azim=args.azim,
        )
    else:
        plot_3d_binary(X3d, labels, out_dir / "plot_by_label.png",
                       f"{base_title}\n(Binary: green=success, red=failure)",
                       elev=args.elev, azim=args.azim)

    # Interactive HTML plots
    if not args.no_html:
        print("\nGenerating interactive HTML plots...")
        plot_3d_interactive_by_task(
            X3d, task_ids, rollouts, rollout_ids,
            out_dir / "plot_by_task_interactive.html",
            f"{base_title} | Colored by Task ID")
        if effective_color_mode == "temporal_gradient":
            progression = compute_failure_progression(rollouts, rollout_ids)
            plot_3d_interactive_temporal_with_green_success(
                X3d, labels, progression, rollouts, rollout_ids, timesteps,
                out_dir / "plot_by_label_interactive.html",
                f"{base_title} | Green-Success + Temporal-Gradient Failure")
        else:
            plot_3d_interactive_binary(
                X3d, labels, rollouts, rollout_ids,
                out_dir / "plot_by_label_interactive.html",
                f"{base_title} | Binary Success/Failure")

    print(f"\nDone. All outputs in: {out_dir.absolute()}")
    print("Tip: Open the *_interactive.html files in any web browser for rotate/zoom/hover.")


if __name__ == "__main__":
    main()