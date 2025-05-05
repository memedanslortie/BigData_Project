import numpy as np

def evaluate(res, gt, k):
    if res.shape[1] < k:
        padded_res = np.zeros((res.shape[0], k), dtype=res.dtype) - 1
        padded_res[:, :res.shape[1]] = res
        res = padded_res
    # calcul du recall
    recall = 0.0
    for i in range(res.shape[0]):
        gt_set = set(gt[i, :k])
        res_set = set(res[i, :k])
        overlap = len(gt_set.intersection(res_set))
        recall += overlap / k

    recall = recall / res.shape[0]
    
    return {"recall@{}".format(k): recall}