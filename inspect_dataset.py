import pickle
import torch

with open("/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx/task_lift_blue_cup/task_lift_blue_cup--ep40--succ0.pkl", "rb") as f:
    data = pickle.load(f)

# Get the first timestep's hidden states
h_t0 = data["hidden_states"][0]  # shape (7, 4096)
print(f"Shape: {h_t0.shape}")

# Compute differences between the 7 vectors
# If they're all the same, they're not 7 action tokens — they're something else
h_float = h_t0.float()
for i in range(7):
    for j in range(i+1, 7):
        diff = (h_float[i] - h_float[j]).abs().mean().item()
        print(f"  Position {i} vs {j}: mean abs diff = {diff:.4f}")