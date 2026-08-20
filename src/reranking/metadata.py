from __future__ import annotations
import numpy as np

def category_consistency_rerank(scores, indices, gallery_df, alpha_visual=0.85):
    """
    Query metadata is unavailable in SOP. Therefore this is NOT an oracle
    query-category match. It uses candidate super_class_id consensus among
    the retrieved Top-N candidates.

    meta_score(candidate) = frequency of its super_class_id in Top-N / N
    final_score = alpha * normalized_visual + (1-alpha) * meta_score
    """
    scores = np.asarray(scores, dtype="float32")
    indices = np.asarray(indices, dtype="int64")
    out_scores = np.empty_like(scores)
    out_indices = np.empty_like(indices)

    meta_by_row = gallery_df.reset_index(drop=True)["super_class_id"].to_numpy()

    for r in range(len(scores)):
        valid = indices[r] >= 0
        ids = indices[r, valid]
        if len(ids) == 0:
            continue
        sc = scores[r, valid]
        cls = meta_by_row[ids]
        values, counts = np.unique(cls, return_counts=True)
        freq = {v: c / len(cls) for v, c in zip(values, counts)}
        meta = np.array([freq[c] for c in cls], dtype="float32")
        vmin, vmax = float(sc.min()), float(sc.max())
        visual = (sc - vmin) / (vmax - vmin + 1e-8)
        final = alpha_visual * visual + (1.0 - alpha_visual) * meta
        order = np.argsort(-final)
        out_indices[r, :len(ids)] = ids[order]
        out_scores[r, :len(ids)] = final[order]
        if len(ids) < indices.shape[1]:
            out_indices[r, len(ids):] = -1
            out_scores[r, len(ids):] = -np.inf
    return out_scores, out_indices
