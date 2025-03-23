import numpy as np
import h5py
import os

def load_dataset(name):
    path = os.path.join("data", f"{name}.hdf5")
    gt_path = os.path.join("data", "ground_truth.hdf5")

    with h5py.File(path, 'r') as f:
        xb = f['train'][:]
        xq = f['test'][:]

    with h5py.File(gt_path, 'r') as f:
        gt = f['neighbors'][:]

    if gt.shape[0] != xq.shape[0]:
        print(f"ground truth truncated: {gt.shape[0]} → {xq.shape[0]}")
        gt = gt[:xq.shape[0]]

    return xb, xq, gt
