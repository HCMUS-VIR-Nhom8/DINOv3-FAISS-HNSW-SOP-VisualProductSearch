from __future__ import annotations
import faiss
import numpy as np

class HNSWRetriever:
    def __init__(self, dim, M=32, ef_construction=200, ef_search=64):
        self.index = faiss.IndexHNSWFlat(dim, int(M), faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = int(ef_construction)
        self.index.hnsw.efSearch = int(ef_search)

    def add(self, embeddings):
        x = np.asarray(embeddings, dtype="float32")
        faiss.normalize_L2(x)
        self.index.add(x)

    def search(self, query_embeddings, k=100):
        q = np.asarray(query_embeddings, dtype="float32").copy()
        faiss.normalize_L2(q)
        scores, ids = self.index.search(q, int(k))
        return scores.astype("float32"), ids.astype("int64")

    def save(self, path):
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path):
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        return obj
