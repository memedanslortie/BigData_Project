import numpy as np

def evaluate(res, gt, k):
    """
    Evaluate recall by checking how many true nearest neighbors are found.
    Uses set overlap approach which is more appropriate for ANN evaluation.
    """
    # Handle the case where res might be shorter than k
    if res.shape[1] < k:
        padded_res = np.zeros((res.shape[0], k), dtype=res.dtype) - 1
        padded_res[:, :res.shape[1]] = res
        res = padded_res
    
    # Calculate recall using set overlap
    recall = 0.0
    for i in range(res.shape[0]):
        # Convert row to set for intersection calculation
        gt_set = set(gt[i, :k])
        res_set = set(res[i, :k])
        # Calculate overlap
        overlap = len(gt_set.intersection(res_set))
        recall += overlap / k
    
    # Average across all queries
    recall = recall / res.shape[0]
    
    return {"recall@{}".format(k): recall}