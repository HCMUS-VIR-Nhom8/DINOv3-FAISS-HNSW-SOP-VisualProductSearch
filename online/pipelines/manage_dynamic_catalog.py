"""CLI quan ly catalog dong (Hinh 2.9) - them san pham vao Dynamic Buffer hoac
go san pham bang Blacklist, KHONG can rebuild Static HNSW.

Chay tu thu muc goc project:
    python -m online.pipelines.manage_dynamic_catalog add --image anh.jpg --product-id 999001
    python -m online.pipelines.manage_dynamic_catalog remove --product-id 12345
"""
import argparse

from common.backbone.dinov3_encoder import DINOv3Encoder
from common.config import load_config
from common.preprocessing.image_io import load_image
from online.retrieval.blacklist import BlacklistSet
from online.retrieval.dynamic_buffer import DynamicBufferIndex


def add_product(cfg: dict, image_path: str, product_id: int):
    encoder = DINOv3Encoder.from_config(cfg["model"])
    dyn = DynamicBufferIndex.load(
        cfg["model"]["embedding_dim"], cfg["dynamic"]["buffer_path"], cfg["dynamic"]["buffer_id_map_path"]
    )

    image = load_image(image_path)
    embedding = encoder.encode([image])
    dyn.add(embedding, [product_id])
    dyn.save(cfg["dynamic"]["buffer_path"], cfg["dynamic"]["buffer_id_map_path"])

    print(f"Da them product_id={product_id}. Buffer hien co {dyn.size()} san pham.")
    if dyn.size() >= cfg["dynamic"]["rebuild_threshold"]:
        print(
            "Buffer da vuot rebuild_threshold - nen chay lai "
            "offline.pipelines.run_offline_indexing de gop buffer vao Static HNSW roi reset buffer."
        )


def remove_product(cfg: dict, product_id: int):
    blacklist = BlacklistSet.load(cfg["blacklist"]["path"])
    blacklist.add(product_id)
    blacklist.save(cfg["blacklist"]["path"])
    print(f"Da them product_id={product_id} vao blacklist.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/online.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p_add = sub.add_parser("add")
    p_add.add_argument("--image", required=True)
    p_add.add_argument("--product-id", type=int, required=True)

    p_remove = sub.add_parser("remove")
    p_remove.add_argument("--product-id", type=int, required=True)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "add":
        add_product(cfg, args.image, args.product_id)
    elif args.command == "remove":
        remove_product(cfg, args.product_id)


if __name__ == "__main__":
    main()
