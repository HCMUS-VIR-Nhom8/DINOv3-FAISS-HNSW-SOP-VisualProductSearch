"""Entry point chu trinh ONLINE BASELINE (Hinh 1.1):
    Query Image -> Preprocessing -> DINOv3 Encoder -> Index HNSW
    -> Top-N ung vien -> Re-ranking -> Top-K ket qua cuoi cung

Khong dung Dynamic Buffer / Blacklist - xem run_online_retrieval_dynamic.py
cho ban day du theo Hinh 2.9.

Chay tu thu muc goc project:
    python -m online.pipelines.run_online_retrieval --image query.jpg --config configs/online.yaml
"""
import argparse

from PIL import Image

from common.backbone.dinov3_encoder import DINOv3Encoder
from common.config import load_config
from common.preprocessing.image_io import load_image
from online.reranking.text_rerank import top_k
from online.retrieval.ann_search import ANNSearcher


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/online.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    encoder = DINOv3Encoder.from_config(cfg["model"])
    searcher = ANNSearcher(
        cfg["paths"]["static_index_path"], cfg["paths"]["static_id_map_path"],
        ef_search=cfg["retrieval"]["ef_search"],
    )

    query_image = load_image(args.image)
    query_vec = encoder.encode([query_image])
    scores, ids = searcher.search(query_vec, top_n=cfg["retrieval"]["top_n_candidates"])
    scores, ids = top_k(scores[0], ids[0], cfg["rerank"]["top_k"])

    for rank, (s, pid) in enumerate(zip(scores, ids), start=1):
        print(f"#{rank:>2}  product_id={pid:<10}  score={s:.4f}")


if __name__ == "__main__":
    main()
