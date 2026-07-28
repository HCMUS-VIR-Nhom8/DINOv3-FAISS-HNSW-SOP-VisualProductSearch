"""Trich Global Embedding cho toan bo catalog bang common.backbone.DINOv3Encoder
(Hinh 1.1, box "Preprocessing -> DINOv3 Encoder" cua chu trinh OFFLINE)."""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
from PIL import Image

from common.backbone.dinov3_encoder import DINOv3Encoder


def extract_catalog_embeddings(
    dataset, encoder: DINOv3Encoder, batch_size: int = 64
) -> Tuple[np.ndarray, np.ndarray]:
    """Duyet toan bo dataset theo batch, tra ve (embeddings [N, D], product_ids [N])."""
    all_embeds: List[np.ndarray] = []
    all_ids: List[int] = []
    batch_imgs: List[Image.Image] = []
    batch_ids: List[int] = []

    for image, item in dataset:
        batch_imgs.append(image)
        batch_ids.append(item.product_id)
        if len(batch_imgs) == batch_size:
            all_embeds.append(encoder.encode(batch_imgs))
            all_ids.extend(batch_ids)
            batch_imgs, batch_ids = [], []

    if batch_imgs:
        all_embeds.append(encoder.encode(batch_imgs))
        all_ids.extend(batch_ids)

    embeddings = np.concatenate(all_embeds, axis=0)
    return embeddings, np.array(all_ids, dtype="int64")
