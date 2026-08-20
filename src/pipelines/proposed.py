from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.sop import resolve_image_path
from src.preprocessing.pipeline import ProposedPreprocessor
from src.models.encoder import DINOv3Encoder
from src.retrieval.hnsw import HNSWRetriever
from src.reranking.metadata import category_consistency_rerank
from src.evaluation.benchmark import process_memory_mb

def run(cfg):
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["data"]["csv"])
    root = Path(cfg["data"]["image_root"])
    query_df = df[df["split"] == cfg["data"]["query_split"]].reset_index(drop=True)
    gallery_df = df[df["split"] == cfg["data"]["gallery_split"]].reset_index(drop=True)

    p = cfg["preprocessing"]
    pre = ProposedPreprocessor(
        p["resize_long_side"], p["target_size"], p["padding_ratio"],
        p["blur_threshold"], p["jpeg_threshold"],
        p["illumination_correction"], p["clahe_clip_limit"],
    )
    enc = DINOv3Encoder(cfg["model"]["name"], cfg["model"]["device"])

    def encode_df(frame):
        all_z = []
        quality = []
        start = time.perf_counter()
        bs = cfg["model"]["batch_size"]
        from src.preprocessing.image import load_rgb
        for i in tqdm(range(0, len(frame), bs), desc="Encoding"):
            imgs, qrows = [], []
            for rel in frame.iloc[i:i+bs]["path"]:
                po = pre(resolve_image_path(root, rel))
                imgs.append(po.image)
                qrows.append((po.blur_score, po.jpeg_score, po.metadata))
            all_z.append(enc.encode(imgs))
            quality.extend(qrows)
        return np.vstack(all_z), time.perf_counter() - start, quality

    rss_before = process_memory_mb()
    gallery_z, gallery_encode_s, gallery_quality = encode_df(gallery_df)
    rss_after_gallery = process_memory_mb()
    query_z, query_encode_s, query_quality = encode_df(query_df)

    np.save(out / cfg["output"]["embeddings"], gallery_z)
    np.save(out / "query_embeddings.npy", query_z)

    r = HNSWRetriever(
        dim=gallery_z.shape[1],
        M=cfg["retrieval"]["M"],
        ef_construction=cfg["retrieval"]["efConstruction"],
        ef_search=cfg["retrieval"]["efSearch"],
    )
    build_start = time.perf_counter()
    r.add(gallery_z)
    build_s = time.perf_counter() - build_start
    r.save(out / cfg["output"]["index"])

    t0 = time.perf_counter()
    scores, ids = r.search(query_z, cfg["retrieval"]["candidate_k"])
    retrieval_s = time.perf_counter() - t0

    if cfg["reranking"]["enabled"]:
        scores, ids = category_consistency_rerank(
            scores, ids, gallery_df,
            alpha_visual=cfg["reranking"]["alpha_visual"]
        )

    np.savez_compressed(out / cfg["output"]["results"], scores=scores, ids=ids)
    from src.evaluation.metrics import evaluate
    metrics = evaluate(ids, query_df.class_id.values, gallery_df.class_id.values, cfg["evaluation"]["ks"])
    index_storage_mb = float((out / cfg["output"]["index"]).stat().st_size / (1024**2))
    metrics.update({
        "gallery_encoding_seconds": gallery_encode_s,
        "query_encoding_seconds": query_encode_s,
        "index_build_seconds": build_s,
        "retrieval_total_seconds": retrieval_s,
        "retrieval_latency_ms_per_query": 1000 * retrieval_s / max(1, len(query_df)),
        "reranking_enabled": cfg["reranking"]["enabled"],
        "index_M": cfg["retrieval"]["M"],
        "gallery_embedding_memory_mb": float(gallery_z.nbytes / (1024**2)),
        "hnsw_index_storage_mb": index_storage_mb,
        "total_retrieval_storage_mb": index_storage_mb,
        "rss_delta_after_gallery_encoding_mb": float(rss_after_gallery - rss_before),
        "efConstruction": cfg["retrieval"]["efConstruction"],
        "efSearch": cfg["retrieval"]["efSearch"],
    })
    (out / cfg["output"]["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"image_id": gallery_df.image_id}).to_csv(out / "gallery_ids.csv", index=False)
    pd.DataFrame({"image_id": query_df.image_id}).to_csv(out / "query_ids.csv", index=False)
    pd.DataFrame(gallery_quality, columns=["blur_score","jpeg_score","quality_flags"]).to_json(
        out / "gallery_quality.json", orient="records"
    )
    return metrics
