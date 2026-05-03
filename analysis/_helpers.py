"""Shared utilities for SAFE latent-space analyses.

Loads OpenVLA-WidowX rollouts and turns them into a flat array of points
ready for t-SNE, distance computation, etc.
"""
import argparse
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm


# ============================================================================
# Argument parsing — shared options used by all analysis scripts
# ============================================================================

def add_shared_args(parser: argparse.ArgumentParser):
    """Add the data-loading / preprocessing arguments common to all analyses."""
    parser.add_argument(
        "--rollout_root",
        type=str,
        default="/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx",
        help="Root folder containing per-task rollout subfolders.",
    )
    parser.add_argument(
        "--token_agg",
        choices=["mean", "first", "last"],
        default="mean",
        help="How to collapse the 7 action-token features into 1 vector per timestep.",
    )
    parser.add_argument(
        "--timestep_mode",
        choices=["per_step", "aggregate", "late_per_step", "late_aggregate"],
        default="per_step",
        help=(
            "How to use the temporal dimension:\n"
            "  per_step: each timestep is a separate point (50 points per rollout)\n"
            "  aggregate: average all timesteps per rollout (1 point per rollout)\n"
            "  late_per_step: only timesteps >= start_step, each separate\n"
            "  late_aggregate: only timesteps >= start_step, averaged"
        ),
    )
    parser.add_argument(
        "--start_step",
        type=int,
        default=40,
        help="Earliest timestep to include in 'late_*' modes (rollouts have 50 steps total).",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        default=None,
        help=(
            "Optional comma-separated list of task IDs to keep, e.g. '4,5'. "
            "If None, use all tasks."
        ),
    )


# ============================================================================
# Data loading
# ============================================================================

def load_all_rollouts(rollout_root: str) -> list[dict]:
    """Load every rollout pkl file in the dataset.

    Returns a list of dicts with keys:
        hidden_states: (50, 7, 4096) float32 numpy array
        label: int (0=success, 1=failure)
        task_id: int
        task_description: str
        episode_idx: int
        filename: str
    """
    root = Path(rollout_root)
    rollouts = []

    pkl_files = sorted(root.rglob("*.pkl"))
    print(f"Loading {len(pkl_files)} rollouts from {root}...")

    for pkl_file in tqdm(pkl_files):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        hs_list = data["hidden_states"]  # list of T tensors of shape (7, 4096) bf16
        # Stack into a single tensor (T, 7, 4096), convert to float32 numpy
        hs = torch.stack(hs_list).float().numpy()  # (T, 7, 4096)

        rollouts.append({
            "hidden_states": hs,
            "label": int(not data["episode_success"]),  # success=False -> label=1 (fail)
            "task_id": int(data["task_id"]),
            "task_description": data["task_description"],
            "episode_idx": int(data["eposide_idx"]),
            "filename": pkl_file.name,
        })

    return rollouts


# ============================================================================
# Preprocessing: turn rollouts into a flat (N, D) point array
# ============================================================================

def aggregate_tokens(hs: np.ndarray, mode: str) -> np.ndarray:
    """Collapse the token dimension. Input shape (T, 7, 4096) -> (T, 4096)."""
    if mode == "mean":
        return hs.mean(axis=1)
    elif mode == "first":
        return hs[:, 0, :]
    elif mode == "last":
        return hs[:, -1, :]
    else:
        raise ValueError(f"Unknown token_agg: {mode}")


def rollouts_to_points(
    rollouts: list[dict],
    token_agg: str,
    timestep_mode: str,
    start_step: int = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn a list of rollouts into a flat (N, 4096) point array.

    Returns:
        points: (N, 4096) float32
        labels: (N,) int — 0=success, 1=fail
        task_ids: (N,) int
        rollout_ids: (N,) int — index into the rollouts list
        timesteps: (N,) int — original timestep within rollout (or -1 if aggregated)
    """
    points_list = []
    labels_list = []
    task_ids_list = []
    rollout_ids_list = []
    timesteps_list = []

    for ri, r in enumerate(rollouts):
        hs = aggregate_tokens(r["hidden_states"], token_agg)  # (T, 4096)
        T = hs.shape[0]

        if timestep_mode == "per_step":
            for t in range(T):
                points_list.append(hs[t])
                labels_list.append(r["label"])
                task_ids_list.append(r["task_id"])
                rollout_ids_list.append(ri)
                timesteps_list.append(t)

        elif timestep_mode == "aggregate":
            points_list.append(hs.mean(axis=0))  # (4096,)
            labels_list.append(r["label"])
            task_ids_list.append(r["task_id"])
            rollout_ids_list.append(ri)
            timesteps_list.append(-1)

        elif timestep_mode == "late_per_step":
            for t in range(start_step, T):
                points_list.append(hs[t])
                labels_list.append(r["label"])
                task_ids_list.append(r["task_id"])
                rollout_ids_list.append(ri)
                timesteps_list.append(t)

        elif timestep_mode == "late_aggregate":
            late = hs[start_step:].mean(axis=0)  # (4096,)
            points_list.append(late)
            labels_list.append(r["label"])
            task_ids_list.append(r["task_id"])
            rollout_ids_list.append(ri)
            timesteps_list.append(-1)
        else:
            raise ValueError(f"Unknown timestep_mode: {timestep_mode}")

    points = np.stack(points_list).astype(np.float32)
    labels = np.array(labels_list, dtype=np.int64)
    task_ids = np.array(task_ids_list, dtype=np.int64)
    rollout_ids = np.array(rollout_ids_list, dtype=np.int64)
    timesteps = np.array(timesteps_list, dtype=np.int64)

    return points, labels, task_ids, rollout_ids, timesteps


def filter_by_tasks(
    rollouts: list[dict], task_filter: Optional[str]
) -> list[dict]:
    """Optionally keep only rollouts whose task_id is in task_filter."""
    if task_filter is None:
        return rollouts
    keep_ids = set(int(x) for x in task_filter.split(","))
    filtered = [r for r in rollouts if r["task_id"] in keep_ids]
    print(f"Task filter {keep_ids}: {len(filtered)}/{len(rollouts)} rollouts kept")
    return filtered


# ============================================================================
# Output management
# ============================================================================

def make_output_dir(script_name: str, base: str = "./analysis/outputs") -> Path:
    """Auto-timestamped output folder: ./analysis/outputs/<script>/<YYYYMMDD-HHMMSS>/"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(base) / script_name / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir.absolute()}")
    return out_dir


# ============================================================================
# Print summary
# ============================================================================

def print_data_summary(rollouts, points, labels, task_ids, args):
    n_rollouts = len(rollouts)
    n_points = len(points)
    n_succ_rollouts = sum(1 for r in rollouts if r["label"] == 0)
    n_fail_rollouts = n_rollouts - n_succ_rollouts
    n_succ_points = (labels == 0).sum()
    n_fail_points = (labels == 1).sum()
    n_tasks = len(set(task_ids.tolist()))

    print()
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Rollouts:   {n_rollouts}  ({n_succ_rollouts} success, {n_fail_rollouts} fail)")
    print(f"  Points:     {n_points}  ({n_succ_points} success, {n_fail_points} fail)")
    print(f"  Tasks:      {n_tasks}")
    print(f"  Feature dim: {points.shape[1]}")
    print(f"  Settings:   token_agg={args.token_agg}, timestep_mode={args.timestep_mode}, start_step={args.start_step}")
    print("=" * 60)
    print()