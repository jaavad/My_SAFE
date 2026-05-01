"""Count successes vs failures per task in the WidowX rollout dataset."""
import os
from pathlib import Path
from collections import defaultdict

rollout_root = Path("/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx")

# Map task_description -> (n_success, n_failure)
counts = defaultdict(lambda: [0, 0])

for task_folder in sorted(rollout_root.iterdir()):
    if not task_folder.is_dir():
        continue
    
    for f in task_folder.glob("*.pkl"):
        # Filename format: <task>--ep<N>--succ<0 or 1>.pkl
        parts = f.stem.split("--")
        succ = int(parts[-1].replace("succ", ""))
        # Use folder name as task ID (because some tasks have multiple folders)
        counts[task_folder.name][succ] += 1

print(f"{'Task folder':45s} {'#Fail':>6s} {'#Success':>10s} {'Total':>7s}")
print("-" * 75)
total_fail = total_succ = 0
for task, (n_fail, n_succ) in sorted(counts.items()):
    print(f"{task:45s} {n_fail:>6d} {n_succ:>10d} {n_fail+n_succ:>7d}")
    total_fail += n_fail
    total_succ += n_succ
print("-" * 75)
print(f"{'TOTAL':45s} {total_fail:>6d} {total_succ:>10d} {total_fail+total_succ:>7d}")