"""Blacklist Set - phan CAI TIEN theo Hinh 2.9.

Roaring Bitmap chua id san pham bi go/an (het hang, vi pham...). FAISS HNSW
khong ho tro xoa hieu qua nen ta LOC ket qua o buoc hau-xu-ly (membership test
cuc nhanh) thay vi xoa that khoi index."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pyroaring import BitMap


class BlacklistSet:
    def __init__(self):
        self.bitmap = BitMap()

    def add(self, product_id: int) -> None:
        self.bitmap.add(product_id)

    def add_many(self, product_ids) -> None:
        self.bitmap.update(product_ids)

    def contains(self, product_id: int) -> bool:
        return product_id in self.bitmap

    def filter_out(self, ids: np.ndarray, scores: np.ndarray):
        """Loai cac id nam trong blacklist khoi 2 mang ids/scores song song (1 query)."""
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
