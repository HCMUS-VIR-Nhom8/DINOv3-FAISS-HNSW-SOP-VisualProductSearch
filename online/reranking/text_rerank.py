"""Multimodal re-ranking: gop ung vien tu nhieu nguon, ket hop diem hinh anh
voi diem khop text (Hinh 2.9, box "Candidate Combination" + "Text Re-ranking Top-K")."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def merge_candidates(
    scores_list: List[np.ndarray], ids_list: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Gop ung vien tu nhieu nguon (Static HNSW + Dynamic Buffer), giu diem cao
    nhat khi trung id (Candidate Combination)."""
    all_scores = np.concatenate(scores_list) if scores_list else np.array([], dtype="float32")
    all_ids = np.concatenate(ids_list) if ids_list else np.array([], dtype="int64")

    best_score: Dict[int, float] = {}
    for s, i in zip(all_scores, all_ids):
        i = int(i)
        if i not in best_score or s > best_score[i]:
            best_score[i] = float(s)

    ids = np.array(list(best_score.keys()), dtype="int64")
    scores = np.array(list(best_score.values()), dtype="float32")
    order = np.argsort(-scores)
    return scores[order], ids[order]


def text_rerank(
    scores: np.ndarray,
    ids: np.ndarray,
    query_text: Optional[str],
    metadata: Optional[Dict[int, str]],
    alpha_visual: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    """score_final = alpha_visual * score_visual + (1 - alpha_visual) * score_text.
    Baseline dung khop tu khoa don gian (bag-of-words overlap); co the thay bang
    BM25 (rank_bm25) hoac cross-encoder khi can chinh xac hon."""
    if not query_text or not metadata:
        return scores, ids

    q_tokens = set(query_text.lower().split())
    text_scores = []
    for pid in ids:
        desc = metadata.get(int(pid), "")
        d_tokens = set(desc.lower().split())
        overlap = len(q_tokens & d_tokens) / max(len(q_tokens), 1)
        text_scores.append(overlap)
    text_scores_arr = np.array(text_scores, dtype="float32")

    final = alpha_visual * scores + (1 - alpha_visual) * text_scores_arr
    order = np.argsort(-final)
    return final[order], ids[order]


def top_k(scores: np.ndarray, ids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    k = min(k, len(ids))
    return scores[:k], ids[:k]
