"""Chay chu trinh OFFLINE (Hinh 1.1): build Static Main HNSW tu toan bo catalog.

Vi du:
    python scripts/01_build_catalog_index.py --config configs/config.yaml
"""
import argparse

import yaml

from vps.data.sop_dataset import SOPDataset
from vps.indexing.static_index import StaticHNSWIndex
from vps.pipeline.baseline_pipeline import OfflineCatalogIndexer
from vps.utils.factory import build_encoder


def main(config_path: str):
    cfg = yaml.safe_load(open(config_path))

    dataset = SOPDataset(root=cfg["dataset"]["root"], list_file=cfg["dataset"]["train_list"])
    encoder = build_encoder(cfg)
    index = StaticHNSWIndex(
        dim=cfg["encoder"]["embedding_dim"],
        m=cfg["index"]["static"]["hnsw_m"],
        ef_construction=cfg["index"]["static"]["ef_construction"],
        ef_search=cfg["index"]["static"]["ef_search"],
    )

    indexer = OfflineCatalogIndexer(encoder, index, batch_size=cfg["encoder"]["batch_size"])
    indexer.build_from_dataset(dataset)
    index.save(cfg["index"]["static"]["path"], cfg["index"]["static"]["id_map_path"])
    print(f"Da build xong Static HNSW voi {index.index.ntotal} san pham.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
