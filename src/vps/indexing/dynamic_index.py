"""
Phan CAI TIEN (Hinh 2.9) de xu ly du lieu catalog thay doi lien tuc (them/xoa san pham)
ma KHONG can rebuild Static HNSW moi lan:

  - DynamicBufferIndex : Flat index (brute-force, chinh xac 100%) chua cac san pham
                          MOI duoc them vao sau lan build Static HNSW gan nhat.
                          Vi buffer nho nen search vet can van du nhanh.
  - BlacklistSet        : tap id san pham bi go/an (het hang, vi pham...), dung Roaring
                          Bitmap de nen va kiem tra membership cuc nhanh - vi FAISS HNSW
                          khong ho tro xoa hieu qua nen ta "an" ket qua o buoc filter thay
                          vi xoa that khoi index.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import faiss
import numpy as np
from pyroaring import BitMap


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
            empty_scores = np.zeros((query.shape[0], 0), dtype="float32")
            empty_ids = np.zeros((query.shape[0], 0), dtype="int64")
            return empty_scores, empty_ids
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


class BlacklistSet:
    """Roaring Bitmap chua cac product_id bi loai khoi ket qua tra ve."""

    def __init__(self):
        self.bitmap = BitMap()

    def add(self, product_id: int) -> None:
        self.bitmap.add(product_id)

    def add_many(self, product_ids: List[int]) -> None:
        self.bitmap.update(product_ids)

    def contains(self, product_id: int) -> bool:
        return product_id in self.bitmap

    def filter_out(self, ids: np.ndarray, scores: np.ndarray):
        """Loai cac id nam trong blacklist khoi 2 mang ids/scores song song (dung cho 1 query)."""
        if len(ids) == 0:
            return scores, ids
        mask = np.array([pid not in self.bitmap for pid in ids])
        return scores[mask], ids[mask]

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            f.write(self.bitmap.serialize())

    @classmethod
    def load(cls, path: str) -> "BlacklistSet":
        obj = cls()
        if Path(path).exists():
            with open(path, "rb") as f:
                obj.bitmap = BitMap.deserialize(f.read())
        return obj
