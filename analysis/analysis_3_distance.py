"""Analysis 3: Distance-based neighborhood analysis of VLA features.

For each point, finds its k-nearest neighbors using 3 distance metrics:
  - Mahalanobis
  - Euclidean
  - Cosine

Then asks: do the neighbors share the same label as the query point?
This is the "do nearby features have similar success/failure status?" question.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EmpiricalCovariance

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
    p = argparse.ArgumentParser(description="Analysis 3: Distance / kNN-agreement analysis.")
    add_shared_args(p)
    p.add_argument("--metrics", type=str, default="mahala,euclid,cosine",
                   help="Comma-separated list of metrics. Choices: mahala, euclid, cosine.")
    p.add_argument("--k", type=int, default=10,
                   help="Number of nearest neighbors to check for label agreement.")
    p.add_argument("--label_mode", choices=["rollout_label", "step_stratified"],
                   default="rollout_label",
                   help=("rollout_label: each point's label = its rollout's label. "
                         "step_stratified: only compare points at similar timesteps "
                         "(only meaningful in per_step modes)."))
    p.add_argument("--step_window", type=int, default=5,
                   help="Window size for step_stratified mode (points within ±W steps).")
    p.add_argument("--pca_first", action="store_true", default=True,
                   help="PCA-reduce to 50 dims before computing distances "
                        "(needed for Mahalanobis if N < D).")
    p.add_argument("--no_pca_first", action="store_false", dest="pca_first")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default=None)
    return p.parse_args()


# ============================================================================
# Distance computations
# ============================================================================

def compute_neighbors_euclidean(X, k):
    """Standard Euclidean k-NN. Returns indices of shape (N, k)."""
    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean", n_jobs=-1)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]  # drop the point itself


def compute_neighbors_cosine(X, k):
    """Cosine k-NN. Returns indices of shape (N, k)."""
    nn = NearestNeighbors(n_neighbors=k+1, metric="cosine", n_jobs=-1)
    nn.fit(X)
    _, idx = nn.kneighbors(X)
    return idx[:, 1:]


def compute_neighbors_mahalanobis(X, k):
    """Mahalanobis k-NN: whitens X using its inverse covariance, then Euclidean kNN.
    
    Note: requires N > D. We assume PCA pre-reduction made D small (e.g., 50).
    """
    cov = EmpiricalCovariance().fit(X)
    # Whitened representation: X_w = X @ inv_cov_sqrt
    # Mahalanobis distance in original space = Euclidean distance after whitening
    L = np.linalg.cholesky(cov.covariance_ + 1e-6 * np.eye(X.shape[1]))
    X_white = np.linalg.solve(L, X.T).T  # (N, D)
    return compute_neighbors_euclidean(X_white, k)


def compute_neighbors(X: np.ndarray, metric: str, k: int) -> np.ndarray:
    if metric == "euclid":
        return compute_neighbors_euclidean(X, k)
    elif metric == "cosine":
        return compute_neighbors_cosine(X, k)
    elif metric == "mahala":
        return compute_neighbors_mahalanobis(X, k)
    else:
        raise ValueError(f"Unknown metric: {metric}")


# ============================================================================
# Agreement scoring
# ============================================================================

def neighbor_agreement(neighbors: np.ndarray, labels: np.ndarray) -> dict:
    """For each point, compute fraction of its neighbors that share its label.
    
    Returns:
        agreement: (N,) float — per-point fraction (0 to 1)
        summary: dict with overall stats
    """
    N, k = neighbors.shape
    own = labels[:, None]               # (N, 1)
    nbr = labels[neighbors]             # (N, k)
    agree = (own == nbr).mean(axis=1)   # (N,)

    # Overall stats
    overall = agree.mean()
    agree_succ = agree[labels == 0].mean() if (labels == 0).any() else float("nan")
    agree_fail = agree[labels == 1].mean() if (labels == 1).any() else float("nan")

    return {
        "per_point": agree,
        "overall_mean": float(overall),
        "succ_mean": float(agree_succ),
        "fail_mean": float(agree_fail),
    }


def neighbor_agreement_step_stratified(
    neighbors: np.ndarray,
    labels: np.ndarray,
    timesteps: np.ndarray,
    window: int = 5,
) -> dict:
    """Like neighbor_agreement, but only count neighbors within ±window timesteps."""
    N, k = neighbors.shape
    agree = np.zeros(N)
    for i in range(N):
        # Filter neighbors to those within ±window of point i's timestep
        mask = np.abs(timesteps[neighbors[i]] - timesteps[i]) <= window
        if mask.sum() == 0:
            agree[i] = float("nan")
        else:
            agree[i] = (labels[neighbors[i][mask]] == labels[i]).mean()

    valid = ~np.isnan(agree)
    overall = agree[valid].mean()
    succ_valid = valid & (labels == 0)
    fail_valid = valid & (labels == 1)
    return {
        "per_point": agree,
        "overall_mean": float(overall),
        "succ_mean": float(agree[succ_valid].mean()) if succ_valid.any() else float("nan"),
        "fail_mean": float(agree[fail_valid].mean()) if fail_valid.any() else float("nan"),
        "n_with_zero_neighbors": int((~valid).sum()),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    # Validate label_mode against timestep_mode
    if args.label_mode == "step_stratified" and args.timestep_mode in ("aggregate", "late_aggregate"):
        raise ValueError(
            "label_mode=step_stratified is meaningless when timesteps are aggregated. "
            "Use --label_mode rollout_label instead."
        )

    # Output directory
    if args.output_dir is None:
        out_dir = make_output_dir("analysis_3_distance")
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

    # PCA pre-reduction (recommended for Mahalanobis stability)
    X = points
    if args.pca_first and X.shape[1] > 50:
        print(f"PCA pre-reduction: {X.shape[1]} -> 50 dims")
        X = PCA(n_components=50, random_state=args.seed).fit_transform(X)

    # Compute and score for each metric
    metrics = [m.strip() for m in args.metrics.split(",")]
    results = {}
    for metric in metrics:
        print(f"\n--- Metric: {metric} ---")
        neighbors = compute_neighbors(X, metric, args.k)
        if args.label_mode == "rollout_label":
            score = neighbor_agreement(neighbors, labels)
        else:
            score = neighbor_agreement_step_stratified(
                neighbors, labels, timesteps, args.step_window
            )
        results[metric] = score
        print(f"  Overall agreement: {score['overall_mean']:.4f}")
        print(f"  Among success points: {score['succ_mean']:.4f}")
        print(f"  Among failure points: {score['fail_mean']:.4f}")

    # Save results CSV
    csv_path = out_dir / "agreement_summary.csv"
    with open(csv_path, "w") as f:
        f.write("metric,overall,succ,fail\n")
        for metric, s in results.items():
            f.write(f"{metric},{s['overall_mean']:.4f},{s['succ_mean']:.4f},{s['fail_mean']:.4f}\n")
    print(f"\nSaved CSV: {csv_path}")

    # Bar plot summary
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics_list = list(results.keys())
    x = np.arange(len(metrics_list))
    width = 0.25

    overall = [results[m]["overall_mean"] for m in metrics_list]
    succ = [results[m]["succ_mean"] for m in metrics_list]
    fail = [results[m]["fail_mean"] for m in metrics_list]

    ax.bar(x - width, overall, width, label="Overall", color="gray")
    ax.bar(x,         succ,    width, label="Success points", color="tab:blue")
    ax.bar(x + width, fail,    width, label="Failure points", color="tab:red")

    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.5, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_list)
    ax.set_ylabel(f"k={args.k} Neighbor agreement (fraction)")
    ax.set_title(
        f"Neighborhood label agreement\n"
        f"timestep_mode={args.timestep_mode}, label_mode={args.label_mode}"
    )
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    plot_path = out_dir / "agreement_bar.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved plot: {plot_path}")

    # Save raw arrays for further analysis
    np.savez(
        out_dir / "raw_results.npz",
        X=X,
        labels=labels,
        task_ids=task_ids,
        timesteps=timesteps,
        **{f"agree_{m}": results[m]["per_point"] for m in metrics_list},
    )
    print(f"\nDone. All outputs in: {out_dir.absolute()}")


if __name__ == "__main__":
    main()