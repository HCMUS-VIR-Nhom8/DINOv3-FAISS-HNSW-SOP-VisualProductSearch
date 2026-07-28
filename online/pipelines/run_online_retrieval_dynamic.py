"""Entry point chu trinh ONLINE CAI TIEN (Hinh 2.9):
    Vector Query -> ANN Parallel Search (Static HNSW + Dynamic Buffer)
    -> Candidate Combination -> Loai bo ID thuoc Blacklist -> Text Re-ranking Top-K

Chay tu thu muc goc project:
    python -m online.pipelines.run_online_retrieval_dynamic \\
        --image query.jpg --config configs/online.yaml --text "tui xach da"
"""
import argparse
from concurrent.futures import ThreadPoolExecutor

from common.backbone.dinov3_encoder import DINOv3Encoder
from common.config import load_config
from common.preprocessing.image_io import load_image
from online.reranking.text_rerank import merge_candidates, text_rerank, top_k
from online.retrieval.ann_search import ANNSearcher
from online.retrieval.blacklist import BlacklistSet
from online.retrieval.dynamic_buffer import DynamicBufferIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--config", default="configs/online.yaml")
    parser.add_argument("--text", default=None, help="Text bo sung de re-rank (tuy chon)")
    args = parser.parse_args()
    cfg = load_config(args.config)

    encoder = DINOv3Encoder.from_config(cfg["model"])
    static_searcher = ANNSearcher(
        cfg["paths"]["static_index_path"], cfg["paths"]["static_id_map_path"],
        ef_search=cfg["retrieval"]["ef_search"],
    )
    dynamic_index = DynamicBufferIndex.load(
        cfg["model"]["embedding_dim"], cfg["dynamic"]["buffer_path"], cfg["dynamic"]["buffer_id_map_path"]
    )
    blacklist = BlacklistSet.load(cfg["blacklist"]["path"])

    query_image = load_image(args.image)
    query_vec = encoder.encode([query_image])
    top_n = cfg["retrieval"]["top_n_candidates"]

    # ANN Parallel Search: faiss.search giai phong GIL nen ThreadPoolExecutor mang lai parallel that
    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_static = ex.submit(static_searcher.search, query_vec, top_n)
        fut_dynamic = ex.submit(dynamic_index.search, query_vec, top_n)
        s_scores, s_ids = fut_static.result()
        d_scores, d_ids = fut_dynamic.result()

    # Candidate Combination
    scores, ids = merge_candidates([s_scores[0], d_scores[0]], [s_ids[0], d_ids[0]])
    # Loai bo ID thuoc Blacklist
    scores, ids = blacklist.filter_out(ids, scores)
    # Text Re-ranking Top-K
    scores, ids = text_rerank(scores, ids, args.text, metadata=None, alpha_visual=cfg["rerank"]["alpha_visual"])
    scores, ids = top_k(scores, ids, cfg["rerank"]["top_k"])

    for rank, (s, pid) in enumerate(zip(scores, ids), start=1):
        print(f"#{rank:>2}  product_id={pid:<10}  score={s:.4f}")


if __name__ == "__main__":
    main()
