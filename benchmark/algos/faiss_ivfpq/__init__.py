import faiss
import numpy as np
import time

class FaissIVFPQ:
    def __init__(self, nlist=100, m=8, nbits=8, nprobe=10, k_reorder=0):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.k_reorder = k_reorder
        self.index = None
        self.xb = None

    def fit(self, xb):
        d = xb.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits)
        self.index.train(xb)
        self.index.add(xb)
        self.index.nprobe = self.nprobe
        self.xb = xb

    def query(self, xq, k):
        start = time.time()
        D, I = self.index.search(xq, max(k, self.k_reorder))  # chercher + que k si reorder
        if self.k_reorder > 0:
            reordered_I = []
            for i in range(xq.shape[0]):
                candidates = self.xb[I[i]]
                dists = np.linalg.norm(candidates - xq[i], axis=1)
                topk = np.argsort(dists)[:k]
                reordered_I.append(I[i][topk])
            I = np.array(reordered_I)
        end = time.time()
        self.last_search_time = end - start
        return I
