"""Analysis 1 (3D, simple two-color):
- Success points = blue
- Failure points = blue if timestep < fail_boundary, red otherwise

Configurable boundary lets you ask: 'when do failure rollouts start to look different?'
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
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

# --- Sharp colors ---
COLOR_BLUE = "#1f77b4"   # success + early failure
COLOR_RED  = "#d62728"   # late failure


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 1 (3D, simple): blue/red with timestep boundary.")
    add_shared_args(p)
    p.add_argument("--method", choices=["tsne", "pca", "umap"], default="tsne")
    p.add_argument("--pca_first", action="store_true", default=False,
                   help="Pre-reduce to 50 dims with PCA before t-SNE. Default OFF.")
    p.add_argument("--feat_skip", type=int, default=1,
                   help="Take every N-th timestep. Default 1 = keep ALL.")
    p.add_argument("--fail_boundary", type=int, default=25,
                   help="In per_step modes: failure points with timestep < fail_boundary "
                        "are colored BLUE (treated as still-OK). Points at timestep >= "
                        "fail_boundary are colored RED. Default 5.")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_html", action="store_true",
                   help="Skip interactive Plotly HTML output.")
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


def build_points(rollouts, token_agg, feat_skip, timestep_mode, start_step):
    feats_list, labels_list, tids_list, rids_list, ts_list = [], [], [], [], []
    for ri, r in enumerate(rollouts):
        hs = aggregate_tokens(r["hidden_states"], token_agg)
        T = hs.shape[0]
        if timestep_mode == "per_step":
            keep_ts = list(range(0, T, feat_skip))
        elif timestep_mode == "aggregate":
            feats_list.append(hs.mean(axis=0)[None, :])
            labels_list.append([r["label"]]); tids_list.append([r["task_id"]])
            rids_list.append([ri]); ts_list.append([-1]); continue
        elif timestep_mode == "late_per_step":
            keep_ts = list(range(start_step, T, feat_skip))
        elif timestep_mode == "late_aggregate":
            feats_list.append(hs[start_step:].mean(axis=0)[None, :])
            labels_list.append([r["label"]]); tids_list.append([r["task_id"]])
            rids_list.append([ri]); ts_list.append([-1]); continue
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


def categorize_points(labels, timesteps, fail_boundary):
    """Returns a 3-way category per point:
       0 = success           -> BLUE
       1 = early-failure     -> BLUE  (failure rollout, timestep < fail_boundary)
       2 = late-failure      -> RED   (failure rollout, timestep >= fail_boundary)
    For aggregated points (timestep == -1), failures all go to category 2 (RED).
    """
    cats = np.zeros(len(labels), dtype=np.int64)
    for i in range(len(labels)):
        if labels[i] == 0:
            cats[i] = 0
        else:
            if timesteps[i] == -1:
                cats[i] = 2  # aggregated failure -> red
            elif timesteps[i] < fail_boundary:
                cats[i] = 1
            else:
                cats[i] = 2
    return cats


# ============================================================================
# Static PNG plots (Matplotlib 3D)
# ============================================================================

def plot_3d_two_color(X3d, cats, save_path, title, fail_boundary, elev=20, azim=-60):
    """Two-color plot: blue (cat 0,1) and red (cat 2)."""
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    blue_mask = (cats == 0) | (cats == 1)
    red_mask = cats == 2
    n_succ = (cats == 0).sum()
    n_early_fail = (cats == 1).sum()
    n_late_fail = red_mask.sum()

    ax.scatter(X3d[blue_mask, 0], X3d[blue_mask, 1], X3d[blue_mask, 2],
               c=COLOR_BLUE,
               label=f"Blue: success ({n_succ}) + early-fail t<{fail_boundary} ({n_early_fail})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.scatter(X3d[red_mask, 0], X3d[red_mask, 1], X3d[red_mask, 2],
               c=COLOR_RED,
               label=f"Red: late-fail t>={fail_boundary} ({n_late_fail})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(markerscale=2.5, fontsize=10, loc="upper left")
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

def plot_3d_interactive_two_color(
    X3d, cats, timesteps, rollouts, rollout_ids, save_path, title, fail_boundary
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML. Run: pip install plotly")
        return

    blue_mask = (cats == 0) | (cats == 1)
    red_mask = cats == 2

    def make_hover(idxs):
        return [
            f"Task {rollouts[rollout_ids[j]]['task_id']}: "
            f"{rollouts[rollout_ids[j]]['task_description']}"
            f"<br>Episode {rollouts[rollout_ids[j]]['episode_idx']}"
            f"<br>{'Success' if rollouts[rollout_ids[j]]['label']==0 else 'Failure'}"
            f"<br>Timestep: {timesteps[j] if timesteps[j] >= 0 else 'aggregated'}"
            f"<br>Category: {['success','early-fail','late-fail'][cats[j]]}"
            for j in idxs
        ]

    fig = go.Figure()

    # Blue points
    blue_idxs = np.where(blue_mask)[0]
    n_succ = (cats == 0).sum()
    n_early_fail = (cats == 1).sum()
    fig.add_trace(go.Scatter3d(
        x=X3d[blue_mask, 0], y=X3d[blue_mask, 1], z=X3d[blue_mask, 2],
        mode="markers",
        marker=dict(size=3, color=COLOR_BLUE, opacity=0.65),
        name=f"Blue: success ({n_succ}) + early-fail ({n_early_fail})",
        text=make_hover(blue_idxs), hoverinfo="text",
    ))

    # Red points
    red_idxs = np.where(red_mask)[0]
    n_late_fail = red_mask.sum()
    fig.add_trace(go.Scatter3d(
        x=X3d[red_mask, 0], y=X3d[red_mask, 1], z=X3d[red_mask, 2],
        mode="markers",
        marker=dict(size=3, color=COLOR_RED, opacity=0.65),
        name=f"Red: late-fail t>={fail_boundary} ({n_late_fail})",
        text=make_hover(red_idxs), hoverinfo="text",
    ))

    fig.update_layout(
        title=f"{title}<br>Boundary: failures with timestep>={fail_boundary} are RED",
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
        out_dir = make_output_dir("analysis_1_tsne_3d_simple")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_all_rollouts(args.rollout_root)
    rollouts = filter_by_tasks(rollouts, args.task_filter)
    rollouts = sorted(rollouts, key=lambda r: (r["task_id"], r["episode_idx"]))

    feats, labels, task_ids, rollout_ids, timesteps = build_points(
        rollouts, args.token_agg, args.feat_skip, args.timestep_mode, args.start_step
    )

    fake_args = argparse.Namespace(
        token_agg=args.token_agg,
        timestep_mode=args.timestep_mode,
        start_step=args.start_step,
    )
    print_data_summary(rollouts, feats, labels, task_ids, fake_args)

    print(f"  feat_skip:     {args.feat_skip}")
    print(f"  fail_boundary: {args.fail_boundary}")
    print(f"  pca_first:     {args.pca_first}")
    print(f"  Reducing to 3D...\n")

    X3d = reduce_to_3d(feats, args)

    cats = categorize_points(labels, timesteps, args.fail_boundary)

    np.savez(
        out_dir / "embedding_3d.npz",
        X3d=X3d, labels=labels, task_ids=task_ids,
        rollout_ids=rollout_ids, timesteps=timesteps,
        cats=cats,
        args=str(vars(args)),
    )

    n_blue = (cats != 2).sum()
    n_red = (cats == 2).sum()
    print(f"  -> Blue points: {n_blue}  | Red points: {n_red}")

    base_title = (
        f"{args.method.upper()} 3D | token={args.token_agg} | "
        f"mode={args.timestep_mode} | feat_skip={args.feat_skip} | "
        f"fail_boundary={args.fail_boundary}"
    )
    if "late" in args.timestep_mode:
        base_title += f" (start={args.start_step})"

    plot_3d_by_task(X3d, task_ids, out_dir / "plot_by_task.png",
                    f"{base_title}\nColored by Task ID",
                    elev=args.elev, azim=args.azim)

    plot_3d_two_color(X3d, cats, out_dir / "plot_by_label.png",
                      f"{base_title}\n(Blue=success+early-fail, Red=late-fail)",
                      args.fail_boundary,
                      elev=args.elev, azim=args.azim)

    if not args.no_html:
        print("\nGenerating interactive HTML plots...")
        plot_3d_interactive_by_task(
            X3d, task_ids, rollouts, rollout_ids,
            out_dir / "plot_by_task_interactive.html",
            f"{base_title} | Colored by Task ID")
        plot_3d_interactive_two_color(
            X3d, cats, timesteps, rollouts, rollout_ids,
            out_dir / "plot_by_label_interactive.html",
            f"{base_title} | Two-Color (Blue/Red)",
            args.fail_boundary)

    print(f"\nDone. All outputs in: {out_dir.absolute()}")
    print("Tip: Open *_interactive.html files in any web browser for rotate/zoom/hover.")


if __name__ == "__main__":
    main()