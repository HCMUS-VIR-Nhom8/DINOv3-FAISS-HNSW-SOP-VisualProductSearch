"""Ham dung chung de khoi tao cac thanh phan tu file config.yaml, tranh lap code giua cac script."""
from __future__ import annotations

from vps.encoders.dinov3_encoder import DINOv3Encoder
from vps.indexing.dynamic_index import BlacklistSet, DynamicBufferIndex
from vps.indexing.static_index import StaticHNSWIndex


def build_encoder(cfg: dict) -> DINOv3Encoder:
    enc_cfg = cfg["encoder"]
    return DINOv3Encoder(
        backend=enc_cfg["backend"],
        model_name=enc_cfg["model_name"],
        torchhub_repo_dir=enc_cfg["torchhub_repo_dir"],
        torchhub_arch=enc_cfg["torchhub_arch"],
        torchhub_weights=enc_cfg["torchhub_weights"],
        device=enc_cfg["device"],
        image_size=enc_cfg["image_size"],
    )


def load_static_index(cfg: dict) -> StaticHNSWIndex:
    idx_cfg = cfg["index"]["static"]
    return StaticHNSWIndex.load(idx_cfg["path"], idx_cfg["id_map_path"], ef_search=idx_cfg["ef_search"])


def load_dynamic_index(cfg: dict) -> DynamicBufferIndex:
    idx_cfg = cfg["index"]["dynamic"]
    return DynamicBufferIndex.load(cfg["encoder"]["embedding_dim"], idx_cfg["buffer_path"], idx_cfg["buffer_id_map_path"])


def load_blacklist(cfg: dict) -> BlacklistSet:
    return BlacklistSet.load(cfg["index"]["blacklist"]["path"])
