"""
Pipeline CAI TIEN - dung theo Hinh 2.9: chay song song Static Main Index (HNSW) va
Dynamic Buffer Index (Flat), gop ung vien (Candidate Combination), loc Blacklist
(Roaring Bitmap), roi Text Re-ranking de ra Top-K ket qua cuoi cung.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

from PIL import Image

from vps.encoders.dinov3_encoder import DINOv3Encoder
from vps.indexing.dynamic_index import BlacklistSet, DynamicBufferIndex
from vps.indexing.static_index import StaticHNSWIndex
from vps.reranking.reranker import merge_candidates, text_rerank, top_k


class ImprovedSearchPipeline:
    def __init__(
        self,
        encoder: DINOv3Encoder,
        static_index: StaticHNSWIndex,
        dynamic_index: DynamicBufferIndex,
        blacklist: BlacklistSet,
        top_n: int = 200,
        top_k_final: int = 20,
    ):
        self.encoder = encoder
        self.static_index = static_index
        self.dynamic_index = dynamic_index
        self.blacklist = blacklist
        self.top_n = top_n
        self.top_k_final = top_k_final

    def search(
        self,
        query_image: Image.Image,
        query_text: Optional[str] = None,
        metadata: Optional[Dict[int, str]] = None,
    ):
        query_vec = self.encoder.encode([query_image])  # Vector Query (q)

        # ANN Parallel Search: tim song song tren Static HNSW va Dynamic Buffer (Flat).
        # faiss.search giai phong GIL nen ThreadPoolExecutor mang lai parallel that.
        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_static = ex.submit(self.static_index.search, query_vec, self.top_n)
            fut_dynamic = ex.submit(self.dynamic_index.search, query_vec, self.top_n)
            s_scores, s_ids = fut_static.result()
            d_scores, d_ids = fut_dynamic.result()

        # Candidate Combination
        scores, ids = merge_candidates([s_scores[0], d_scores[0]], [s_ids[0], d_ids[0]])

        # Loai bo ID thuoc Blacklist
        scores, ids = self.blacklist.filter_out(ids, scores)

        # Text Re-ranking Top-K
        scores, ids = text_rerank(scores, ids, query_text, metadata)
        scores, ids = top_k(scores, ids, self.top_k_final)
        return scores, ids
