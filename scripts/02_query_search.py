"""Chay chu trinh ONLINE baseline (Hinh 1.1) cho 1 anh truy van.

Vi du:
    python scripts/02_query_search.py --query path/to/anh.jpg
"""
import argparse

import yaml
from PIL import Image

from vps.pipeline.baseline_pipeline import BaselineSearchPipeline
from vps.utils.factory import build_encoder, load_static_index


def main(config_path: str, query_image_path: str):
    cfg = yaml.safe_load(open(config_path))

    encoder = build_encoder(cfg)
    index = load_static_index(cfg)
    pipeline = BaselineSearchPipeline(
        encoder, index, top_n=cfg["search"]["top_n_candidates"], top_k_final=cfg["search"]["top_k_final"]
    )

    query_image = Image.open(query_image_path).convert("RGB")
    scores, ids = pipeline.search(query_image)
    for rank, (s, pid) in enumerate(zip(scores, ids), start=1):
        print(f"#{rank:>2}  product_id={pid:<10}  score={s:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--query", required=True, help="Duong dan anh truy van")
    args = parser.parse_args()
    main(args.config, args.query)
