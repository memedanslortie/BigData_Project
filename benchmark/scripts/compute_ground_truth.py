import faiss
import numpy as np
import h5py
import os

DATASET = "fashion-mnist-784-euclidean.hdf5"
OUTPUT = "ground_truth.hdf5"
K = 10

path = os.path.join("data", DATASET)
with h5py.File(path, 'r') as f:
    xb = f['train'][:]
    xq = f['test'][:]

index = faiss.IndexFlatL2(xb.shape[1])
index.add(xb)
D, I = index.search(xq, K)

with h5py.File(os.path.join("data", OUTPUT), 'w') as f:
    f.create_dataset("neighbors", data=I)

print("Ground truth enregistré dans data/ground_truth.hdf5")
