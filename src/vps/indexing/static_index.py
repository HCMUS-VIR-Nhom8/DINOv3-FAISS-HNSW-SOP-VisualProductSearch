"""
Static Main Index (HNSW) - dung cho chu trinh OFFLINE (Hinh 1.1):
    Catalog Images -> Preprocessing -> DINOv3 Encoder -> Static Main HNSW
va duoc truy van trong chu trinh ONLINE (ca baseline lan improved).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class StaticHNSWIndex:
    def __init__(self, dim: int, m: int = 32, ef_construction: int = 200, ef_search: int = 128):
        self.dim = dim
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = ef_construction
        self.index.hnsw.efSearch = ef_search
        self.id_map: Optional[np.ndarray] = None  # vi tri noi bo FAISS -> product_id thuc te

    def build(self, embeddings: np.ndarray, product_ids: np.ndarray) -> None:
        """embeddings: [N, D] float32 da L2-normalize; product_ids: [N] id san pham goc."""
        assert embeddings.shape[1] == self.dim, "Sai so chieu embedding so voi index"
        self.index.add(embeddings)
        self.id_map = product_ids.astype("int64")

    def search(self, query: np.ndarray, top_n: int):
        """query: [B, D]. Tra ve (scores [B,N], product_ids [B,N]) da map ve id goc."""
        scores, positions = self.index.search(query, top_n)
        ids = self.id_map[positions]
        return scores, ids

    def save(self, index_path: str, id_map_path: str) -> None:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        np.save(id_map_path, self.id_map)

    @classmethod
    def load(cls, index_path: str, id_map_path: str, ef_search: int = 128) -> "StaticHNSWIndex":
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(index_path)
        obj.dim = obj.index.d
        obj.index.hnsw.efSearch = ef_search
        obj.id_map = np.load(id_map_path)
        return obj
