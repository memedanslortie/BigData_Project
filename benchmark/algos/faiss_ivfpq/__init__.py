import faiss
import numpy as np
import time

class FaissIVFPQ:
    def __init__(self, nlist=100, m=8, nbits=8, nprobe=10):
        self.nlist = nlist
        self.m = m
        self.nbits = nbits
        self.nprobe = nprobe
        self.index = None

    def fit(self, xb):
        d = xb.shape[1]
        quantizer = faiss.IndexFlatL2(d)
        self.index = faiss.IndexIVFPQ(quantizer, d, self.nlist, self.m, self.nbits)
        self.index.train(xb)
        self.index.add(xb)
        self.index.nprobe = self.nprobe

    def query(self, xq, k):
        start = time.time()
        D, I = self.index.search(xq, k)
        end = time.time()
        self.last_search_time = end - start 
        return I