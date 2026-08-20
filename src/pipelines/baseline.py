from __future__ import annotations
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.sop import attach_split, resolve_image_path
from src.preprocessing.pipeline import BaselinePreprocessor
from src.models.encoder import ResNet50Encoder
from src.retrieval.exact import ExactCosineRetriever
from src.evaluation.benchmark import process_memory_mb

def run(cfg):
    out = Path(cfg["output"]["dir"]); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(cfg["data"]["csv"])
    root = Path(cfg["data"]["image_root"])
    query_df = df[df["split"] == cfg["data"]["query_split"]].reset_index(drop=True)
    gallery_df = df[df["split"] == cfg["data"]["gallery_split"]].reset_index(drop=True)

    pre = BaselinePreprocessor(cfg["preprocessing"]["resize"][0])
    enc = ResNet50Encoder(cfg["model"]["device"], cfg["model"]["pretrained"])

    def encode_df(frame):
        all_z = []
        start = time.perf_counter()
        bs = cfg["model"]["batch_size"]
        for i in tqdm(range(0, len(frame), bs), desc="Encoding"):
            pil = []
            for rel in frame.iloc[i:i+bs]["path"]:
                from src.preprocessing.image import load_rgb
                pil.append(load_rgb(resolve_image_path(root, rel)))
            all_z.append(enc.encode(pil))
        return np.vstack(all_z), time.perf_counter() - start

    rss_before = process_memory_mb()
    gallery_z, gallery_encode_s = encode_df(gallery_df)
    rss_after_gallery = process_memory_mb()
    query_z, query_encode_s = encode_df(query_df)
    np.save(out / cfg["output"]["embeddings"], gallery_z)
    np.save(out / "query_embeddings.npy", query_z)

    retriever = ExactCosineRetriever(gallery_z)
    k = cfg["retrieval"]["top_k"]
    t0 = time.perf_counter()
    scores, ids = retriever.search(query_z, k)
    retrieval_s = time.perf_counter() - t0

    np.savez_compressed(out / cfg["output"]["results"], scores=scores, ids=ids)
    from src.evaluation.metrics import evaluate
    metrics = evaluate(ids, query_df.class_id.values, gallery_df.class_id.values, cfg["evaluation"]["ks"])
    metrics["gallery_encoding_seconds"] = gallery_encode_s
    metrics["gallery_embedding_memory_mb"] = float(gallery_z.nbytes / (1024**2))
    metrics["rss_delta_after_gallery_encoding_mb"] = float(rss_after_gallery - rss_before)
    metrics["total_retrieval_storage_mb"] = metrics["gallery_embedding_memory_mb"]
    metrics["query_encoding_seconds"] = query_encode_s
    metrics["retrieval_total_seconds"] = retrieval_s
    metrics["retrieval_latency_ms_per_query"] = 1000 * retrieval_s / max(1, len(query_df))
    (out / cfg["output"]["metrics"]).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({"image_id": gallery_df.image_id}).to_csv(out / "gallery_ids.csv", index=False)
    pd.DataFrame({"image_id": query_df.image_id}).to_csv(out / "query_ids.csv", index=False)
    return metrics
