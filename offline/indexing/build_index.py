"""Build Static Main Index (HNSW) tu embedding da trich xuat
(Hinh 1.1, box "Static Main HNSW" cua chu trinh OFFLINE).

Dung ham thuan (function) thay vi class de de dispatch theo config.index.family
va de mo rong sang ho khac (vd IVFPQ) khi catalog scale len sau nay."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import faiss
import numpy as np


def build_hnsw_index(
    embeddings: np.ndarray, m: int = 32, ef_construction: int = 200,
    metric=faiss.METRIC_INNER_PRODUCT,
) -> faiss.Index:
    """Khong can train() - do thi duoc xay truc tiep khi add(). Ho tro them vector
    moi ("on-the-fly") sau nay ma khong can rebuild toan bo - xem docs/method_faiss_hnsw.md."""
    d = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(d, m, metric)
    index.hnsw.efConstruction = ef_construction
    index.add(embeddings)
    return index


def save_index(index: faiss.Index, product_ids: np.ndarray, index_path: str, id_map_path: str) -> None:
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, index_path)
    np.save(id_map_path, product_ids.astype("int64"))


def load_index(index_path: str, id_map_path: str) -> Tuple[faiss.Index, np.ndarray]:
    index = faiss.read_index(index_path)
    id_map = np.load(id_map_path)
    return index, id_map
