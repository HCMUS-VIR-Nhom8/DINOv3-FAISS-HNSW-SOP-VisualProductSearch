"""Entry point chu trinh OFFLINE (Hinh 1.1):
    Catalog Images -> Preprocessing -> DINOv3 Encoder -> Static Main HNSW

Chay tu thu muc goc project:
    python -m offline.pipelines.run_offline_indexing --config configs/offline.yaml
"""
import argparse

from common.backbone.dinov3_encoder import DINOv3Encoder
from common.config import load_config
from offline.feature_extraction.extract_embeddings import extract_catalog_embeddings
from offline.feature_extraction.sop_dataset import SOPDataset
from offline.indexing.build_index import build_hnsw_index, save_index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/offline.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)

    dataset = SOPDataset(root=cfg["dataset"]["root"], list_file=cfg["dataset"]["catalog_list"])
    encoder = DINOv3Encoder.from_config(cfg["model"])

    embeddings, product_ids = extract_catalog_embeddings(
        dataset, encoder, batch_size=cfg["model"]["batch_size"]
    )

    index_cfg = cfg["index"]
    if index_cfg["family"] == "hnsw":
        index = build_hnsw_index(embeddings, m=index_cfg["hnsw_m"], ef_construction=index_cfg["ef_construction"])
    else:
        raise ValueError(f"Chua ho tro index family: {index_cfg['family']}")

    save_index(index, product_ids, cfg["paths"]["static_index_path"], cfg["paths"]["static_id_map_path"])
    print(f"Da build xong Static HNSW voi {index.ntotal} san pham -> {cfg['paths']['static_index_path']}")


if __name__ == "__main__":
    main()
