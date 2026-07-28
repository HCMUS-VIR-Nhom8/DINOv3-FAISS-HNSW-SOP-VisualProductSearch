"""Test logic cho build_index / ANNSearcher / DynamicBufferIndex / BlacklistSet /
text_rerank - khong can model DINOv3 that."""
import numpy as np

from offline.indexing.build_index import build_hnsw_index
from online.reranking.text_rerank import merge_candidates, top_k
from online.retrieval.blacklist import BlacklistSet
from online.retrieval.dynamic_buffer import DynamicBufferIndex


def test_build_hnsw_index_ntotal_khop_input():
    dim = 16
    embeddings = np.random.rand(50, dim).astype("float32")
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = build_hnsw_index(embeddings)
    assert index.ntotal == 50

    query = embeddings[:3]
    _, positions = index.search(query, 5)
    # truy van dung vector da index thi ket qua top-1 phai la chinh no (vi tri 0..2)
    assert (positions[:, 0] == np.arange(3)).all()


def test_dynamic_buffer_and_blacklist():
    dim = 8
    dyn = DynamicBufferIndex(dim=dim)
    embeddings = np.random.rand(3, dim).astype("float32")
    dyn.add(embeddings, [1, 2, 3])
    assert dyn.size() == 3

    bl = BlacklistSet()
    bl.add(2)
    scores = np.array([0.9, 0.8, 0.7], dtype="float32")
    ids = np.array([1, 2, 3], dtype="int64")
    filtered_scores, filtered_ids = bl.filter_out(ids, scores)
    assert 2 not in filtered_ids
    assert len(filtered_ids) == 2


def test_merge_and_top_k():
    scores_a = np.array([0.9, 0.5], dtype="float32")
    ids_a = np.array([1, 2], dtype="int64")
    scores_b = np.array([0.95, 0.4], dtype="float32")  # id=1 trung, diem cao hon
    ids_b = np.array([1, 3], dtype="int64")

    scores, ids = merge_candidates([scores_a, scores_b], [ids_a, ids_b])
    assert len(ids) == 3
    assert ids[0] == 1 and scores[0] == 0.95

    scores, ids = top_k(scores, ids, k=2)
    assert len(ids) == 2
