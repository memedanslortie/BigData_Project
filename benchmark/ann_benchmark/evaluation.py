import numpy as np

def evaluate(res, gt, k):
    total = res.shape[0] * k
    correct = (res == gt[:, :k]).sum()
    recall = correct / total
    return {"recall@{}".format(k): recall}

