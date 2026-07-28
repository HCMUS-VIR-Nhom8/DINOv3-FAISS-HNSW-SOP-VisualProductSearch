"""Chay chu trinh ONLINE cai tien (Hinh 2.9):
Static HNSW + Dynamic Buffer + Blacklist + Text Re-ranking.

Vi du:
    python scripts/05_query_search_improved.py --query path/to/anh.jpg --text "ao thun nam"
"""
import argparse

import yaml
from PIL import Image

from vps.pipeline.improved_pipeline import ImprovedSearchPipeline
from vps.utils.factory import build_encoder, load_blacklist, load_dynamic_index, load_static_index


def main(config_path: str, query_image_path: str, query_text: str):
    cfg = yaml.safe_load(open(config_path))

    encoder = build_encoder(cfg)
    static_index = load_static_index(cfg)
    dynamic_index = load_dynamic_index(cfg)
    blacklist = load_blacklist(cfg)

    pipeline = ImprovedSearchPipeline(
        encoder,
        static_index,
        dynamic_index,
        blacklist,
        top_n=cfg["search"]["top_n_candidates"],
        top_k_final=cfg["search"]["top_k_final"],
    )

    query_image = Image.open(query_image_path).convert("RGB")
    scores, ids = pipeline.search(query_image, query_text=query_text)
    for rank, (s, pid) in enumerate(zip(scores, ids), start=1):
        print(f"#{rank:>2}  product_id={pid:<10}  score={s:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--text", default=None, help="Text bo sung de re-rank (tuy chon)")
    args = parser.parse_args()
    main(args.config, args.query, args.text)
