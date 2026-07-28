"""Them 1 san pham moi vao Dynamic Buffer Index - KHONG can rebuild Static HNSW (Hinh 2.9).

Vi du:
    python scripts/03_add_product.py --image path/to/anh.jpg --product-id 999001
"""
import argparse

import yaml
from PIL import Image

from vps.utils.factory import build_encoder, load_dynamic_index


def main(config_path: str, image_path: str, product_id: int):
    cfg = yaml.safe_load(open(config_path))
    encoder = build_encoder(cfg)
    dyn = load_dynamic_index(cfg)

    image = Image.open(image_path).convert("RGB")
    embedding = encoder.encode([image])
    dyn.add(embedding, [product_id])
    dyn.save(cfg["index"]["dynamic"]["buffer_path"], cfg["index"]["dynamic"]["buffer_id_map_path"])

    print(f"Da them product_id={product_id}. Buffer hien co {dyn.size()} san pham.")
    if dyn.size() >= cfg["index"]["dynamic"]["rebuild_threshold"]:
        print(
            "Buffer da vuot nguong rebuild_threshold - nen chay lai "
            "01_build_catalog_index.py de gop buffer vao Static HNSW roi reset buffer."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--image", required=True)
    parser.add_argument("--product-id", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.image, args.product_id)
