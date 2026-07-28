"""Go 1 san pham khoi ket qua tim kiem bang cach them vao Blacklist (Hinh 2.9).
Khong xoa that khoi HNSW vi FAISS HNSW khong ho tro xoa hieu qua.

Vi du:
    python scripts/04_remove_product.py --product-id 12345
"""
import argparse

import yaml

from vps.utils.factory import load_blacklist


def main(config_path: str, product_id: int):
    cfg = yaml.safe_load(open(config_path))
    blacklist = load_blacklist(cfg)
    blacklist.add(product_id)
    blacklist.save(cfg["index"]["blacklist"]["path"])
    print(f"Da them product_id={product_id} vao blacklist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--product-id", type=int, required=True)
    args = parser.parse_args()
    main(args.config, args.product_id)
