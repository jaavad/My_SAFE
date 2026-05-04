"""Analysis 1 — UMAP visualization (2D or 3D), full options.

Modes for failure coloring:
  - binary:   success=blue, failure=red (no time info)
  - boundary: success=blue, failure colored blue if t<fail_boundary else red (sharp)
  - gradual:  success=blue, failure colored blue->red gradient starting at fail_boundary

Examples:
  # 2D UMAP, aggregate, binary colors
  python analysis/analysis_1_umap.py --n_dims 2 --timestep_mode aggregate

  # 3D UMAP, per_step, sharp boundary at timestep 10
  python analysis/analysis_1_umap.py --n_dims 3 --timestep_mode per_step \
      --fail_color_mode boundary --fail_boundary 10

  # 3D UMAP, per_step, gradual gradient starting at timestep 10
  python analysis/analysis_1_umap.py --n_dims 3 --timestep_mode per_step \
      --fail_color_mode gradual --fail_boundary 10

  # 2D UMAP, late timesteps only, with PCA pre-reduction
  python analysis/analysis_1_umap.py --n_dims 2 --timestep_mode late_per_step \
      --start_step 30 --pca_first
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    add_shared_args,
    load_all_rollouts,
    filter_by_tasks,
    make_output_dir,
    print_data_summary,
    aggregate_tokens,
)

# Sharp colors
COLOR_BLUE = "#1f77b4"
COLOR_RED  = "#d62728"


def parse_args():
    p = argparse.ArgumentParser(description="UMAP visualization (2D or 3D) with full options.")
    add_shared_args(p)
    p.add_argument("--n_dims", type=int, choices=[2, 3], default=2,
                   help="Dimensionality of the UMAP output: 2 or 3.")
    p.add_argument("--pca_first", action="store_true", default=False,
                   help="Pre-reduce features to 50 dims with PCA before UMAP. Default OFF.")
    p.add_argument("--feat_skip", type=int, default=1,
                   help="Take every N-th timestep. Default 1 = keep ALL.")
    p.add_argument("--fail_color_mode", choices=["binary", "boundary", "gradual"],
                   default="boundary",
                   help="binary:   success=blue, failure=red.\n"
                        "boundary: success=blue, failure=blue if t<fail_boundary else red (sharp).\n"
                        "gradual:  success=blue, failure fades blue->red starting at fail_boundary.")
    p.add_argument("--fail_boundary", type=int, default=5,
                   help="Timestep at which failure points start being red "
                        "(in 'boundary' or 'gradual' modes).")
    # UMAP hyperparams
    p.add_argument("--n_neighbors", type=int, default=15,
                   help="UMAP n_neighbors. Smaller = more local structure (10-50 typical).")
    p.add_argument("--min_dist", type=float, default=0.1,
                   help="UMAP min_dist. Smaller = tighter clusters (0.0-0.99 typical).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--no_html", action="store_true",
                   help="Skip the interactive Plotly HTML output (3D only).")
    p.add_argument("--elev", type=float, default=20.0)
    p.add_argument("--azim", type=float, default=-60.0)
    return p.parse_args()


# ============================================================================
# Data prep
# ============================================================================

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


def reduce_with_umap(X, n_dims, pca_first, n_neighbors, min_dist, seed):
    """UMAP with optional PCA pre-reduction."""
    if pca_first and X.shape[1] > 50:
        print(f"PCA pre-reduction: {X.shape[1]} -> 50 dims")
        X = PCA(n_components=50, random_state=seed).fit_transform(X)

    import umap
    print(f"Running UMAP on {X.shape} -> {n_dims}D "
          f"(n_neighbors={n_neighbors}, min_dist={min_dist})...")
    reducer = umap.UMAP(
        n_components=n_dims,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
        verbose=True,
    )
    return reducer.fit_transform(X)


# ============================================================================
# Color computation (3 modes)
# ============================================================================

def compute_colors(labels, timesteps, mode, fail_boundary, T_max=50):
    """Returns:
        cat: array of int — 0=success/blue, 1=failure but blue, 2=failure red
        scalar: array of float in [0, 1] — used by 'gradual' mode for cmap.
                For other modes scalar is ignored.

    Mode logic:
      - binary:   success -> cat=0; failure -> cat=2 (no scalar use)
      - boundary: success -> cat=0; failure with t<fail_boundary -> cat=1; else cat=2
      - gradual:  success -> cat=0; failure -> cat='gradient', scalar in [0,1]
                  scalar = 0 for failure points with t <= fail_boundary
                  scalar = (t - fail_boundary) / (T_max - 1 - fail_boundary), else clipped to [0,1]
    """
    n = len(labels)
    cat = np.zeros(n, dtype=np.int64)        # 0 = blue (success or early-fail), 2 = red, 3 = gradient
    scalar = np.zeros(n, dtype=np.float32)   # 0..1 for gradient

    if mode == "binary":
        for i in range(n):
            cat[i] = 2 if labels[i] == 1 else 0
        return cat, scalar

    if mode == "boundary":
        for i in range(n):
            if labels[i] == 0:
                cat[i] = 0
            else:
                # Failure
                if timesteps[i] == -1:
                    # aggregated -> always red
                    cat[i] = 2
                elif timesteps[i] < fail_boundary:
                    cat[i] = 0  # treat as blue (early failure)
                else:
                    cat[i] = 2
        return cat, scalar

    if mode == "gradual":
        denom = max(T_max - 1 - fail_boundary, 1)
        for i in range(n):
            if labels[i] == 0:
                cat[i] = 0
                scalar[i] = 0.0
            else:
                if timesteps[i] == -1:
                    cat[i] = 3
                    scalar[i] = 1.0  # aggregated failure -> fully red
                elif timesteps[i] <= fail_boundary:
                    cat[i] = 0  # early failure -> still blue
                    scalar[i] = 0.0
                else:
                    cat[i] = 3
                    s = (timesteps[i] - fail_boundary) / denom
                    scalar[i] = float(np.clip(s, 0.0, 1.0))
        return cat, scalar

    raise ValueError(f"Unknown fail_color_mode: {mode}")


# ============================================================================
# Static plotting
# ============================================================================

def plot_2d_two_color(X2d, cat, save_path, title, fail_boundary, fail_color_mode):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    blue_mask = cat == 0
    red_mask = cat == 2
    ax.scatter(X2d[blue_mask, 0], X2d[blue_mask, 1],
               c=COLOR_BLUE, label=f"Blue (n={blue_mask.sum()})",
               s=12, alpha=0.6, edgecolors="none")
    ax.scatter(X2d[red_mask, 0], X2d[red_mask, 1],
               c=COLOR_RED, label=f"Red (n={red_mask.sum()})",
               s=12, alpha=0.6, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_gradient(X2d, cat, scalar, save_path, title):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    blue_mask = cat == 0  # success + early-failure
    grad_mask = cat == 3
    ax.scatter(X2d[blue_mask, 0], X2d[blue_mask, 1],
               c=COLOR_BLUE, label=f"Blue: success+early-fail (n={blue_mask.sum()})",
               s=12, alpha=0.6, edgecolors="none")
    sc = ax.scatter(X2d[grad_mask, 0], X2d[grad_mask, 1],
                    c=scalar[grad_mask], cmap="coolwarm",
                    vmin=0.0, vmax=1.0,
                    s=12, alpha=0.7, edgecolors="none",
                    label=f"Gradient: late-fail (n={grad_mask.sum()})")
    ax.set_title(title)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Failure progression (0=just past boundary, 1=very late)")
    ax.legend(markerscale=2.5, fontsize=10, loc="best")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_2d_by_task(X2d, task_ids, save_path, title):
    unique_tasks = sorted(set(task_ids.tolist()))
    cmap = mpl.cm.get_cmap("tab10", len(unique_tasks))
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=[cmap(i)], label=f"Task {tid}",
                   s=10, alpha=0.6, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=2.5, loc="best", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_3d_two_color(X3d, cat, save_path, title, fail_boundary, fail_color_mode, elev, azim):
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    blue_mask = cat == 0
    red_mask = cat == 2
    ax.scatter(X3d[blue_mask, 0], X3d[blue_mask, 1], X3d[blue_mask, 2],
               c=COLOR_BLUE, label=f"Blue (n={blue_mask.sum()})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.scatter(X3d[red_mask, 0], X3d[red_mask, 1], X3d[red_mask, 2],
               c=COLOR_RED, label=f"Red (n={red_mask.sum()})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(markerscale=2.5, fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_3d_gradient(X3d, cat, scalar, save_path, title, elev, azim):
    fig = plt.figure(figsize=(11, 9), dpi=150)
    ax = fig.add_subplot(111, projection="3d")
    blue_mask = cat == 0
    grad_mask = cat == 3
    ax.scatter(X3d[blue_mask, 0], X3d[blue_mask, 1], X3d[blue_mask, 2],
               c=COLOR_BLUE, label=f"Blue: success+early-fail (n={blue_mask.sum()})",
               s=8, alpha=0.6, edgecolors="none", depthshade=True)
    sc = ax.scatter(X3d[grad_mask, 0], X3d[grad_mask, 1], X3d[grad_mask, 2],
                    c=scalar[grad_mask], cmap="coolwarm",
                    vmin=0.0, vmax=1.0,
                    s=8, alpha=0.7, edgecolors="none", depthshade=True,
                    label=f"Gradient: late-fail (n={grad_mask.sum()})")
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2"); ax.set_zlabel("Dim 3")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label("Failure progression (0=at boundary, 1=very late)")
    ax.legend(markerscale=2.5, fontsize=10, loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"Saved: {save_path}")


def plot_3d_by_task(X3d, task_ids, save_path, title, elev, azim):
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
# Interactive HTML (3D only — for 2D use the static PNG)
# ============================================================================

def plot_3d_interactive_two_color(
    X3d, cat, timesteps, rollouts, rollout_ids, save_path, title, fail_boundary
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML. Run: pip install plotly")
        return
    blue_mask = cat == 0
    red_mask = cat == 2

    def make_hover(idxs):
        return [
            f"Task {rollouts[rollout_ids[j]]['task_id']}: "
            f"{rollouts[rollout_ids[j]]['task_description']}"
            f"<br>Episode {rollouts[rollout_ids[j]]['episode_idx']}"
            f"<br>{'Success' if rollouts[rollout_ids[j]]['label']==0 else 'Failure'}"
            f"<br>Timestep: {timesteps[j] if timesteps[j] >= 0 else 'aggregated'}"
            for j in idxs
        ]

    fig = go.Figure()
    blue_idxs = np.where(blue_mask)[0]
    red_idxs = np.where(red_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=X3d[blue_mask, 0], y=X3d[blue_mask, 1], z=X3d[blue_mask, 2],
        mode="markers",
        marker=dict(size=3, color=COLOR_BLUE, opacity=0.65),
        name=f"Blue (n={blue_mask.sum()})",
        text=make_hover(blue_idxs), hoverinfo="text",
    ))
    fig.add_trace(go.Scatter3d(
        x=X3d[red_mask, 0], y=X3d[red_mask, 1], z=X3d[red_mask, 2],
        mode="markers",
        marker=dict(size=3, color=COLOR_RED, opacity=0.65),
        name=f"Red (n={red_mask.sum()})",
        text=make_hover(red_idxs), hoverinfo="text",
    ))
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        width=1100, height=900,
    )
    fig.write_html(str(save_path))
    print(f"Saved interactive HTML: {save_path}")


def plot_3d_interactive_gradient(
    X3d, cat, scalar, timesteps, rollouts, rollout_ids, save_path, title
):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Skipping HTML.")
        return
    blue_mask = cat == 0
    grad_mask = cat == 3

    def make_hover(idxs, with_scalar=False):
        out = []
        for j in idxs:
            base = (
                f"Task {rollouts[rollout_ids[j]]['task_id']}: "
                f"{rollouts[rollout_ids[j]]['task_description']}"
                f"<br>Episode {rollouts[rollout_ids[j]]['episode_idx']}"
                f"<br>{'Success' if rollouts[rollout_ids[j]]['label']==0 else 'Failure'}"
                f"<br>Timestep: {timesteps[j] if timesteps[j] >= 0 else 'aggregated'}"
            )
            if with_scalar:
                base += f"<br>Color value: {scalar[j]:.2f}"
            out.append(base)
        return out

    fig = go.Figure()
    blue_idxs = np.where(blue_mask)[0]
    grad_idxs = np.where(grad_mask)[0]
    fig.add_trace(go.Scatter3d(
        x=X3d[blue_mask, 0], y=X3d[blue_mask, 1], z=X3d[blue_mask, 2],
        mode="markers",
        marker=dict(size=3, color=COLOR_BLUE, opacity=0.65),
        name=f"Blue: success+early-fail (n={blue_mask.sum()})",
        text=make_hover(blue_idxs), hoverinfo="text",
    ))
    fig.add_trace(go.Scatter3d(
        x=X3d[grad_mask, 0], y=X3d[grad_mask, 1], z=X3d[grad_mask, 2],
        mode="markers",
        marker=dict(
            size=3, opacity=0.7,
            color=scalar[grad_mask], colorscale="RdBu_r",
            cmin=0.0, cmax=1.0,
            colorbar=dict(title="Late-failure<br>progression"),
        ),
        name=f"Gradient: late-fail (n={grad_mask.sum()})",
        text=make_hover(grad_idxs, with_scalar=True), hoverinfo="text",
    ))
    fig.update_layout(
        title=title,
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
        out_dir = make_output_dir(f"analysis_1_umap_{args.n_dims}d")
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

    print(f"  feat_skip:       {args.feat_skip}")
    print(f"  fail_color_mode: {args.fail_color_mode}")
    print(f"  fail_boundary:   {args.fail_boundary}")
    print(f"  n_dims:          {args.n_dims}")
    print(f"  pca_first:       {args.pca_first}")
    print(f"  UMAP n_neighbors: {args.n_neighbors}, min_dist: {args.min_dist}\n")

    Xnd = reduce_with_umap(
        feats, args.n_dims, args.pca_first,
        args.n_neighbors, args.min_dist, args.seed,
    )

    cat, scalar = compute_colors(
        labels, timesteps, args.fail_color_mode, args.fail_boundary, T_max=50,
    )

    np.savez(
        out_dir / "embedding.npz",
        X=Xnd, labels=labels, task_ids=task_ids,
        rollout_ids=rollout_ids, timesteps=timesteps,
        cat=cat, scalar=scalar,
        args=str(vars(args)),
    )

    base_title = (
        f"UMAP {args.n_dims}D | token={args.token_agg} | "
        f"mode={args.timestep_mode} | feat_skip={args.feat_skip} | "
        f"color={args.fail_color_mode}"
    )
    if args.fail_color_mode in ("boundary", "gradual"):
        base_title += f" (boundary={args.fail_boundary})"
    if "late" in args.timestep_mode:
        base_title += f" (start={args.start_step})"
    if args.pca_first:
        base_title += " | PCA-first"

    if args.n_dims == 2:
        plot_2d_by_task(Xnd, task_ids, out_dir / "plot_by_task.png",
                        f"{base_title}\nColored by Task ID")
        if args.fail_color_mode == "gradual":
            plot_2d_gradient(Xnd, cat, scalar, out_dir / "plot_by_label.png",
                             f"{base_title}\n(Gradient failure)")
        else:
            plot_2d_two_color(Xnd, cat, out_dir / "plot_by_label.png",
                              f"{base_title}\n(Two-color)",
                              args.fail_boundary, args.fail_color_mode)
    else:
        plot_3d_by_task(Xnd, task_ids, out_dir / "plot_by_task.png",
                        f"{base_title}\nColored by Task ID",
                        args.elev, args.azim)
        if args.fail_color_mode == "gradual":
            plot_3d_gradient(Xnd, cat, scalar, out_dir / "plot_by_label.png",
                             f"{base_title}\n(Gradient failure)",
                             args.elev, args.azim)
        else:
            plot_3d_two_color(Xnd, cat, out_dir / "plot_by_label.png",
                              f"{base_title}\n(Two-color)",
                              args.fail_boundary, args.fail_color_mode,
                              args.elev, args.azim)

        if not args.no_html:
            print("\nGenerating interactive HTML plots...")
            plot_3d_interactive_by_task(
                Xnd, task_ids, rollouts, rollout_ids,
                out_dir / "plot_by_task_interactive.html",
                f"{base_title} | Colored by Task ID")
            if args.fail_color_mode == "gradual":
                plot_3d_interactive_gradient(
                    Xnd, cat, scalar, timesteps, rollouts, rollout_ids,
                    out_dir / "plot_by_label_interactive.html",
                    f"{base_title} | Gradient")
            else:
                plot_3d_interactive_two_color(
                    Xnd, cat, timesteps, rollouts, rollout_ids,
                    out_dir / "plot_by_label_interactive.html",
                    f"{base_title} | Two-Color",
                    args.fail_boundary)

    print(f"\nDone. All outputs in: {out_dir.absolute()}")


if __name__ == "__main__":
    main()