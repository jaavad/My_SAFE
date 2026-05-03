"""Check the timestep length of every rollout in the dataset."""
import pickle
from pathlib import Path
from collections import defaultdict, Counter

rollout_root = Path("/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx")

# Per-task length info
lengths_per_task = defaultdict(list)
all_lengths = []

n_files = 0
for task_folder in sorted(rollout_root.iterdir()):
    if not task_folder.is_dir():
        continue
    
    for pkl_file in task_folder.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        T = len(data["hidden_states"])
        task_desc = data.get("task_description", task_folder.name)
        success = data.get("episode_success", None)
        
        lengths_per_task[task_desc].append((T, success, pkl_file.name))
        all_lengths.append(T)
        n_files += 1

print(f"=== Total rollouts inspected: {n_files} ===\n")

# Overall length distribution
length_counter = Counter(all_lengths)
print("=== Distribution of rollout lengths (overall) ===")
print(f"{'Length T':>10s} {'Count':>8s} {'Pct':>7s}")
for T, count in sorted(length_counter.items()):
    pct = 100 * count / n_files
    print(f"{T:>10d} {count:>8d} {pct:>6.1f}%")

print(f"\n  Min length: {min(all_lengths)}")
print(f"  Max length: {max(all_lengths)}")
print(f"  Mean length: {sum(all_lengths)/len(all_lengths):.1f}")

# Per-task length distribution
print("\n=== Per-task length distribution ===")
print(f"{'Task':40s} {'#':>4s} {'min':>5s} {'max':>5s} {'mean':>6s} {'unique_lengths':s}")
for task, entries in sorted(lengths_per_task.items()):
    Ts = [e[0] for e in entries]
    unique = sorted(set(Ts))
    unique_str = str(unique) if len(unique) <= 5 else f"{len(unique)} different values"
    print(f"{task:40s} {len(Ts):>4d} {min(Ts):>5d} {max(Ts):>5d} {sum(Ts)/len(Ts):>6.1f}  {unique_str}")

# Check whether successful and failed rollouts have different length distributions
print("\n=== Length distribution by success/failure ===")
succ_lengths = []
fail_lengths = []
for task, entries in lengths_per_task.items():
    for T, success, _ in entries:
        if success is True:
            succ_lengths.append(T)
        elif success is False:
            fail_lengths.append(T)

if succ_lengths:
    print(f"  Successful: n={len(succ_lengths)}, min={min(succ_lengths)}, max={max(succ_lengths)}, mean={sum(succ_lengths)/len(succ_lengths):.1f}")
if fail_lengths:
    print(f"  Failed:     n={len(fail_lengths)}, min={min(fail_lengths)}, max={max(fail_lengths)}, mean={sum(fail_lengths)/len(fail_lengths):.1f}")