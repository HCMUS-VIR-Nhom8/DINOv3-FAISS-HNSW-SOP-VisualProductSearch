"""Load Static Main Index (HNSW, xay boi offline/pipelines/run_offline_indexing.py)
va thuc hien ANN search top-N cho anh query (Hinh 1.1 va 2.9, phan "Static Main HNSW").

Chi muc phai duoc build boi CUNG mot checkpoint DINOv3 nhu common/backbone/dinov3_encoder.py
dang dung online, de dam bao embedding space nhat quan offline/online."""
import faiss
import numpy as np


class ANNSearcher:
    def __init__(self, index_path: str, id_map_path: str, ef_search: int = 128):
        self.index = faiss.read_index(index_path)
        self.id_map = np.load(id_map_path)
        if hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = ef_search

    def search(self, query_embedding: np.ndarray, top_n: int = 200):
        """query_embedding: [B, D], da qua DUNG DINOv3Encoder nhu luc index.
        Tra ve (scores [B, top_n], product_ids [B, top_n])."""
        scores, positions = self.index.search(query_embedding, top_n)
        ids = self.id_map[positions]
        return scores, ids
