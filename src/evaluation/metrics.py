from __future__ import annotations
import numpy as np

def relevant_mask(result_ids, query_class_id, gallery_class_ids):
    return np.asarray(gallery_class_ids)[np.asarray(result_ids)] == query_class_id

def recall_at_k(results, query_class_ids, gallery_class_ids, k):
    vals = []
    gallery_class_ids = np.asarray(gallery_class_ids)
    for ids, qc in zip(results[:, :k], query_class_ids):
        valid = ids >= 0
        if not valid.any():
            vals.append(0.0)
            continue
        retrieved = gallery_class_ids[ids[valid]]
        relevant_total = max(1, int((gallery_class_ids == qc).sum()))
        vals.append(float((retrieved == qc).sum()) / relevant_total)
    return float(np.mean(vals))

def average_precision(ids, query_class_id, gallery_class_ids):
    ids = np.asarray(ids)
    valid = ids >= 0
    ids = ids[valid]
    rel = (np.asarray(gallery_class_ids)[ids] == query_class_id).astype(np.int32)
    total_rel = int((np.asarray(gallery_class_ids) == query_class_id).sum())
    if total_rel == 0:
        return 0.0
    hit = 0
    precision_sum = 0.0
    for rank, is_rel in enumerate(rel, start=1):
        if is_rel:
            hit += 1
            precision_sum += hit / rank
    return precision_sum / total_rel

def mean_average_precision(results, query_class_ids, gallery_class_ids):
    return float(np.mean([
        average_precision(ids, qc, gallery_class_ids)
        for ids, qc in zip(results, query_class_ids)
    ]))

def evaluate(results, query_class_ids, gallery_class_ids, ks):
    out = {f"recall@{k}": recall_at_k(results, query_class_ids, gallery_class_ids, k) for k in ks}
    out["mAP"] = mean_average_precision(results, query_class_ids, gallery_class_ids)
    return out
