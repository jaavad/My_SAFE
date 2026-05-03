"""Analysis 1: 2D dimensionality reduction visualization of OpenVLA-WidowX features.

Generates plots like Figure 1 of the SAFE paper:
  - Plot A: t-SNE colored by success/failure
  - Plot B: t-SNE colored by task ID
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Make sure we can import _helpers from same folder
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import (
    add_shared_args,
    load_all_rollouts,
    rollouts_to_points,
    filter_by_tasks,
    make_output_dir,
    print_data_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 1: 2D visualization of VLA features.")
    add_shared_args(p)
    p.add_argument("--method", choices=["tsne", "pca"], default="tsne",
                   help="Dimensionality reduction method.")
    p.add_argument("--pca_first", action="store_true", default=True,
                   help="Pre-reduce to 50 dims with PCA before t-SNE (recommended for speed).")
    p.add_argument("--no_pca_first", action="store_false", dest="pca_first",
                   help="Disable PCA pre-reduction.")
    p.add_argument("--perplexity", type=float, default=30.0,
                   help="t-SNE perplexity hyperparameter.")
    p.add_argument("--seed", type=int, default=0,
                   help="Random seed for t-SNE.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="If set, override the auto-timestamped output dir.")
    return p.parse_args()


def reduce_to_2d(X: np.ndarray, args) -> np.ndarray:
    """Project (N, D) features to (N, 2)."""
    if args.method == "pca":
        print("Running PCA -> 2D ...")
        return PCA(n_components=2, random_state=args.seed).fit_transform(X)

    # t-SNE path
    if args.pca_first and X.shape[1] > 50:
        print(f"Running PCA pre-reduction: {X.shape[1]} -> 50 dims ...")
        X = PCA(n_components=50, random_state=args.seed).fit_transform(X)

    print(f"Running t-SNE on {X.shape} (perplexity={args.perplexity}). This may take a while ...")
    tsne = TSNE(
        n_components=2,
        perplexity=args.perplexity,
        random_state=args.seed,
        init="pca",
        learning_rate="auto",
        verbose=1,
    )
    return tsne.fit_transform(X)


def plot_by_label(X2d, labels, save_path, title):
    """Scatter plot colored by success/failure."""
    fig, ax = plt.subplots(figsize=(10, 8))
    succ_mask = labels == 0
    fail_mask = labels == 1
    ax.scatter(X2d[succ_mask, 0], X2d[succ_mask, 1],
               c="tab:blue", label="Success (y=0)", s=4, alpha=0.4)
    ax.scatter(X2d[fail_mask, 0], X2d[fail_mask, 1],
               c="tab:red", label="Failure (y=1)", s=4, alpha=0.4)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_by_task(X2d, task_ids, save_path, title):
    """Scatter plot colored by task ID."""
    unique_tasks = sorted(set(task_ids.tolist()))
    cmap = plt.cm.get_cmap("tab10", len(unique_tasks))
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   c=[cmap(i)], label=f"Task {tid}", s=4, alpha=0.4)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_title(title)
    ax.legend(markerscale=3, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    args = parse_args()

    # Output directory
    if args.output_dir is None:
        out_dir = make_output_dir("analysis_1_tsne")
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Load and preprocess
    rollouts = load_all_rollouts(args.rollout_root)
    rollouts = filter_by_tasks(rollouts, args.task_filter)

    points, labels, task_ids, rollout_ids, timesteps = rollouts_to_points(
        rollouts, args.token_agg, args.timestep_mode, args.start_step
    )
    print_data_summary(rollouts, points, labels, task_ids, args)

    # Reduce to 2D
    X2d = reduce_to_2d(points, args)

    # Save the 2D coordinates
    np.savez(
        out_dir / "embedding.npz",
        X2d=X2d, labels=labels, task_ids=task_ids,
        rollout_ids=rollout_ids, timesteps=timesteps,
        args=str(vars(args)),
    )

    # Make and save plots
    base_title = (
        f"{args.method.upper()} | token_agg={args.token_agg} | "
        f"timestep_mode={args.timestep_mode}"
    )
    if "late" in args.timestep_mode:
        base_title += f" (start_step={args.start_step})"

    plot_by_label(X2d, labels, out_dir / "plot_by_label.png",
                  f"{base_title}\nColored by Success/Failure")
    plot_by_task(X2d, task_ids, out_dir / "plot_by_task.png",
                 f"{base_title}\nColored by Task ID")

    print(f"\nDone. All outputs in: {out_dir.absolute()}")


if __name__ == "__main__":
    main()