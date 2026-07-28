"""Dynamic Buffer Index (Flat) - phan CAI TIEN theo Hinh 2.9.

Chua cac san pham MOI duoc them vao sau lan build Static HNSW gan nhat, tranh
phai rebuild HNSW (chi phi cao) moi lan co san pham moi. Vi buffer nho nen
search vet can (brute-force, chinh xac 100%) van du nhanh."""
from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np


class DynamicBufferIndex:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.id_map: List[int] = []

    def add(self, embeddings: np.ndarray, product_ids: List[int]) -> None:
        self.index.add(embeddings)
        self.id_map.extend(product_ids)

    def search(self, query: np.ndarray, top_n: int):
        if self.index.ntotal == 0:
            return (np.zeros((query.shape[0], 0), dtype="float32"),
                    np.zeros((query.shape[0], 0), dtype="int64"))
        top_n = min(top_n, self.index.ntotal)
        scores, positions = self.index.search(query, top_n)
        ids = np.array(self.id_map, dtype="int64")[positions]
        return scores, ids

    def size(self) -> int:
        return self.index.ntotal

    def reset(self) -> None:
        """Goi sau khi da gop (merge) toan bo buffer vao Static HNSW."""
        self.index.reset()
        self.id_map = []

    def save(self, buffer_path: str, id_map_path: str) -> None:
        Path(buffer_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, buffer_path)
        np.save(id_map_path, np.array(self.id_map, dtype="int64"))

    @classmethod
    def load(cls, dim: int, buffer_path: str, id_map_path: str) -> "DynamicBufferIndex":
        obj = cls(dim)
        if Path(buffer_path).exists():
            obj.index = faiss.read_index(buffer_path)
            obj.id_map = np.load(id_map_path).tolist()
        return obj
