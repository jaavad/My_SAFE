"""Analysis 1: 2D dimensionality reduction visualization, matching the SAFE paper.

Mirrors paper's scripts/visualize_features.py but keeps all timesteps:
- Plain t-SNE on raw 4096-dim features (no PCA pre-reduction by default)
- Paper-style temporal gradient coloring (success=blue, failure=blue->red over time)
- All timesteps kept by default (feat_skip=1)
- In aggregate modes, automatically falls back to binary coloring
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    add_shared_args,
    load_all_rollouts,
    rollouts_to_points,
    filter_by_tasks,
    make_output_dir,
    print_data_summary,
    aggregate_tokens,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 1: 2D visualization of VLA features.")
    add_shared_args(p)
    p.add_argument("--method", choices=["tsne", "pca", "umap"], default="tsne")
    p.add_argument("--pca_first", action="store_true", default=False,
                   help="Pre-reduce to 50 dims with PCA before t-SNE. "
                        "Default OFF to match paper. Turn on for ~5x speed.")
    p.add_argument("--feat_skip", type=int, default=1,
                   help="Take every N-th timestep before t-SNE. Default 1 = keep ALL.")
    p.add_argument("--color_mode", choices=["binary", "temporal_gradient"],
                   default="temporal_gradient",
                   help="binary: blue=success, red=failure. "
                        "temporal_gradient: success=blue, failure=blue->red over time. "
                        "(Falls back to binary in aggregate modes.)")
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


def reduce_to_2d(X: np.ndarray, args) -> np.ndarray:
    if args.method == "pca":
        print("Running PCA -> 2D ...")
        return PCA(n_components=2, random_state=args.seed).fit_transform(X)

    if args.method == "umap":
        import umap
        print(f"Running UMAP on {X.shape} ...")
        return umap.UMAP(n_components=2, random_state=args.seed).fit_transform(X)

    if args.pca_first and X.shape[1] > 50:
        print(f"Optional PCA pre-reduction: {X.shape[1]} -> 50 dims")
        X = PCA(n_components=50, random_state=args.seed).fit_transform(X)

    print(f"Running t-SNE on {X.shape} (perplexity={args.perplexity}). May take a while ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        verbose=1,
    )
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
            labels_list.append([r["label"]])
            tids_list.append([r["task_id"]])
            rids_list.append([ri])
            ts_list.append([-1])
            continue
        elif timestep_mode == "late_per_step":
            keep_ts = list(range(start_step, T, feat_skip))
        elif timestep_mode == "late_aggregate":
            feats_list.append(hs[start_step:].mean(axis=0)[None, :])
            labels_list.append([r["label"]])
            tids_list.append([r["task_id"]])
            rids_list.append([ri])
            ts_list.append([-1])
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


def compute_paper_color(rollouts, rollout_ids, timesteps):
    """Paper's coloring: success -> 0, failure -> linspace(0, 1) over its kept timesteps."""
    colors = np.zeros(len(rollout_ids), dtype=np.float32)
    for ri, r in enumerate(rollouts):
        mask = rollout_ids == ri
        n = int(mask.sum())
        if n == 0:
            continue
        if r["label"] == 1:  # failure
            if n == 1:
                # Only 1 point in this failed rollout (aggregate mode) -> set to 1.0 (full red)
                colors[mask] = 1.0
            else:
                colors[mask] = np.linspace(0, 1, n, dtype=np.float32)
        else:  # success
            colors[mask] = 0.0
    return colors


def plot_by_label_binary(X2d, labels, save_path, title):
    """Sharp blue / red colors for binary success/failure."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    succ_mask = labels == 0
    fail_mask = labels == 1
    ax.scatter(X2d[succ_mask, 0], X2d[succ_mask, 1],
               c="#1f77b4", label=f"Success (n={succ_mask.sum()})",
               s=10, alpha=0.6, edgecolors="none")
    ax.scatter(X2d[fail_mask, 0], X2d[fail_mask, 1],
               c="#d62728", label=f"Failure (n={fail_mask.sum()})",
               s=10, alpha=0.6, edgecolors="none")
    ax.set_title(title)
    ax.legend(markerscale=2.5, fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_by_label_temporal(X2d, colors, save_path, title):
    """Paper-style: success and early failure -> blue; late failure -> red.
    
    Crucially we lock vmin=0, vmax=1 so the colorbar uses the FULL coolwarm range."""
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sc = ax.scatter(
        X2d[:, 0], X2d[:, 1],
        c=colors, cmap="coolwarm",
        vmin=0.0, vmax=1.0,            # <-- THE CRUCIAL FIX: lock the color range
        s=8, alpha=0.6, edgecolors="none",
    )
    ax.set_title(title + "\nBlue = success / early-failure   |   Red = late-failure")
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Failure progression (0 = success / early, 1 = late failure)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_by_task(X2d, task_ids, save_path, title):
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


def main():
    args = parse_args()

    if args.output_dir is None:
        out_dir = make_output_dir("analysis_1_tsne")
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

    # Smart auto-fallback: in aggregate modes, temporal_gradient -> binary
    effective_color_mode = args.color_mode
    if args.color_mode == "temporal_gradient" and args.timestep_mode in ("aggregate", "late_aggregate"):
        print(f"  NOTE: timestep_mode={args.timestep_mode} has only 1 point per rollout.")
        print(f"        Falling back to binary coloring (temporal gradient meaningless here).\n")
        effective_color_mode = "binary"

    print(f"  feat_skip:        {args.feat_skip}")
    print(f"  pca_first:        {args.pca_first}")
    print(f"  color_mode:       {args.color_mode}{' -> binary (auto)' if effective_color_mode != args.color_mode else ''}")
    print(f"  Reducing to 2D...\n")

    X2d = reduce_to_2d(feats, args)

    np.savez(
        out_dir / "embedding.npz",
        X2d=X2d, labels=labels, task_ids=task_ids,
        rollout_ids=rollout_ids, timesteps=timesteps,
        args=str(vars(args)),
    )

    base_title = (
        f"{args.method.upper()} | token={args.token_agg} | "
        f"mode={args.timestep_mode} | feat_skip={args.feat_skip}"
    )
    if "late" in args.timestep_mode:
        base_title += f" (start={args.start_step})"

    plot_by_task(X2d, task_ids, out_dir / "plot_by_task.png",
                 f"{base_title}\nColored by Task ID")

    if effective_color_mode == "temporal_gradient":
        colors = compute_paper_color(rollouts, rollout_ids, timesteps)
        plot_by_label_temporal(X2d, colors, out_dir / "plot_by_label.png",
                               f"{base_title}\n(Paper-style temporal gradient)")
    else:
        plot_by_label_binary(X2d, labels, out_dir / "plot_by_label.png",
                             f"{base_title}\n(Binary success/failure)")

    print(f"\nDone. All outputs in: {out_dir.absolute()}")


if __name__ == "__main__":
    main()