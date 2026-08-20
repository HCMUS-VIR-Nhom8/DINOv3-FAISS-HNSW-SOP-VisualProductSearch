from __future__ import annotations
import numpy as np
import time

class ExactCosineRetriever:
    def __init__(self, gallery_embeddings):
        self.gallery = np.asarray(gallery_embeddings, dtype="float32")
        self.gallery = self.gallery / np.maximum(
            np.linalg.norm(self.gallery, axis=1, keepdims=True), 1e-12
        )

    def search(self, query_embeddings, k=100):
        q = np.asarray(query_embeddings, dtype="float32")
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        scores = q @ self.gallery.T
        k = min(k, self.gallery.shape[0])
        idx = np.argpartition(-scores, kth=k-1, axis=1)[:, :k]
        row = np.arange(len(q))[:, None]
        order = np.argsort(-scores[row, idx], axis=1)
        idx = idx[row, order]
        dist = scores[row, idx]
        return dist.astype("float32"), idx.astype("int64")
