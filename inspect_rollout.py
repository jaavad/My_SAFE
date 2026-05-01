"""Inspect one OpenVLA-WidowX rollout pickle file."""
import pickle
import numpy as np
from pathlib import Path

# Pick any rollout pkl file
rollout_path = "/home/javad/Safe_VLA/My_SAFE/SAFE_datasets/openvla_widowx/task_lift_blue_cup/task_lift_blue_cup--ep40--succ0.pkl"

with open(rollout_path, "rb") as f:
    data = pickle.load(f)

print(f"Top-level type: {type(data).__name__}")
print()

def describe(obj, indent=0, key=""):
    """Recursively describe a nested structure."""
    pad = "  " * indent
    if isinstance(obj, dict):
        print(f"{pad}{key}: dict with {len(obj)} keys")
        for k, v in obj.items():
            describe(v, indent + 1, k)
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{key}: {type(obj).__name__} of length {len(obj)}")
        if len(obj) > 0:
            describe(obj[0], indent + 1, "[0]")
    elif isinstance(obj, np.ndarray):
        print(f"{pad}{key}: ndarray, shape={obj.shape}, dtype={obj.dtype}")
    elif hasattr(obj, "shape"):  # torch tensor or similar
        print(f"{pad}{key}: {type(obj).__name__}, shape={obj.shape}, dtype={getattr(obj, 'dtype', '?')}")
    else:
        # Scalar / string / other
        s = str(obj)
        if len(s) > 80:
            s = s[:80] + "..."
        print(f"{pad}{key}: {type(obj).__name__} = {s}")

if isinstance(data, dict):
    describe(data, key="ROOT")
else:
    print(f"Not a dict. It's a {type(data).__name__}: {data}")