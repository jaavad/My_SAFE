"""
Check whether each rollout pkl contains:
  (A) ONE feature tensor that gets aggregated on the fly, OR
  (B) THREE pre-computed feature tensors (first, last, mean)
"""
import pickle
import torch
from pathlib import Path

rollout_path = Path("/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx/task_lift_blue_cup/task_lift_blue_cup--ep40--succ0.pkl")

with open(rollout_path, "rb") as f:
    data = pickle.load(f)

print(f"=== ALL keys in this rollout file ===")
print(f"Keys: {list(data.keys())}")
print()

# Check for any keys containing "first", "last", "mean", "agg"
suspicious_keys = [k for k in data.keys() if any(s in str(k).lower() for s in ['first', 'last', 'mean', 'agg', 'feat'])]
print(f"Keys related to aggregation: {suspicious_keys}")
print()

# Recursively check what's in hidden_states
hs = data["hidden_states"]
print(f"=== hidden_states structure ===")
print(f"Type: {type(hs).__name__}")
print(f"Length: {len(hs)} (number of timesteps)")
print(f"First element shape: {hs[0].shape}")
print(f"First element dtype: {hs[0].dtype}")
print()

# Now confirm: do any timesteps have a different structure?
# (Maybe earlier timesteps store different aggregations?)
shapes = set()
for i, h in enumerate(hs):
    shapes.add(tuple(h.shape))
print(f"Unique shapes across all 50 timesteps: {shapes}")
print(f"→ {'ALL same shape' if len(shapes)==1 else 'MIXED shapes (suspicious!)'}")
print()

# Verify the SAFE code's expectation by simulating the three aggregation methods
h_t0 = hs[0].float()  # (7, 4096) at timestep 0
print(f"=== Simulating three aggregations on timestep 0 ===")
agg_first = h_t0[0]                      # token_idx_rel=0.0
agg_last  = h_t0[-1]                     # token_idx_rel=1.0
agg_mean  = h_t0.mean(dim=0)             # token_idx_rel=mean

print(f"  'first' (h_t0[0]): shape={agg_first.shape}, mean={agg_first.mean():.4f}, std={agg_first.std():.4f}")
print(f"  'last'  (h_t0[6]): shape={agg_last.shape}, mean={agg_last.mean():.4f}, std={agg_last.std():.4f}")
print(f"  'mean'  (avg):     shape={agg_mean.shape}, mean={agg_mean.mean():.4f}, std={agg_mean.std():.4f}")
print()
print(f"Difference 'first' vs 'last': {(agg_first - agg_last).abs().mean():.4f}")
print(f"Difference 'first' vs 'mean': {(agg_first - agg_mean).abs().mean():.4f}")
print(f"Difference 'last'  vs 'mean': {(agg_last  - agg_mean).abs().mean():.4f}")
print()
print("=== CONCLUSION ===")
print("If the differences above are NON-ZERO, then 'first'/'last'/'mean' produce")
print("genuinely different feature vectors from the SAME stored data.")
print("If they were zero, then it would mean the data is already pre-aggregated,")
print("which would make the aggregation choice meaningless.")