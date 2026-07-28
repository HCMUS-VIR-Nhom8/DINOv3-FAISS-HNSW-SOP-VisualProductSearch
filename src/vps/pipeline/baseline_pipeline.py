"""
Pipeline BASELINE - dung theo Hinh 1.1 (chua co dynamic buffer / blacklist).
"""
from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image

from vps.encoders.dinov3_encoder import DINOv3Encoder
from vps.indexing.static_index import StaticHNSWIndex
from vps.reranking.reranker import top_k


class OfflineCatalogIndexer:
    """Chu trinh OFFLINE: Catalog Images -> Preprocessing -> DINOv3 Encoder -> Static Main HNSW."""

    def __init__(self, encoder: DINOv3Encoder, index: StaticHNSWIndex, batch_size: int = 64):
        self.encoder = encoder
        self.index = index
        self.batch_size = batch_size

    def build_from_dataset(self, dataset) -> StaticHNSWIndex:
        all_embeds, all_ids = [], []
        batch_imgs: List[Image.Image] = []
        batch_ids: List[int] = []

        for image, item in dataset:
            batch_imgs.append(image)
            batch_ids.append(item.image_id)
            if len(batch_imgs) == self.batch_size:
                all_embeds.append(self.encoder.encode(batch_imgs))
                all_ids.extend(batch_ids)
                batch_imgs, batch_ids = [], []

        if batch_imgs:
            all_embeds.append(self.encoder.encode(batch_imgs))
            all_ids.extend(batch_ids)

        embeddings = np.concatenate(all_embeds, axis=0)
        self.index.build(embeddings, np.array(all_ids))
        return self.index


class BaselineSearchPipeline:
    """Chu trinh ONLINE (baseline): Query Image -> Preprocessing -> DINOv3 Encoder ->
    Index HNSW -> Top-N ung vien -> Re-ranking -> Top-K ket qua cuoi cung."""

    def __init__(self, encoder: DINOv3Encoder, index: StaticHNSWIndex, top_n: int = 200, top_k_final: int = 20):
        self.encoder = encoder
        self.index = index
        self.top_n = top_n
        self.top_k_final = top_k_final

    def search(self, query_image: Image.Image):
        query_vec = self.encoder.encode([query_image])
        scores, ids = self.index.search(query_vec, self.top_n)
        scores, ids = top_k(scores[0], ids[0], self.top_k_final)
        return scores, ids
