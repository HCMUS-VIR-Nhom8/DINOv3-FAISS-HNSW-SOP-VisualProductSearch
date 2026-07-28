"""FastAPI service expose endpoint /search cho Online Retrieval (Hinh 2.9 day du).

Chay: uvicorn online.api.app:app --reload --port 8000
"""
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from fastapi import FastAPI, UploadFile
from PIL import Image

from common.backbone.dinov3_encoder import DINOv3Encoder
from common.config import load_config
from online.reranking.text_rerank import merge_candidates, text_rerank, top_k
from online.retrieval.ann_search import ANNSearcher
from online.retrieval.blacklist import BlacklistSet
from online.retrieval.dynamic_buffer import DynamicBufferIndex

app = FastAPI(title="Visual Product Search Service")

_cfg = load_config("configs/online.yaml")
_encoder = DINOv3Encoder.from_config(_cfg["model"])
_static_searcher = ANNSearcher(
    _cfg["paths"]["static_index_path"], _cfg["paths"]["static_id_map_path"], ef_search=_cfg["retrieval"]["ef_search"]
)
_dynamic_index = DynamicBufferIndex.load(
    _cfg["model"]["embedding_dim"], _cfg["dynamic"]["buffer_path"], _cfg["dynamic"]["buffer_id_map_path"]
)
_blacklist = BlacklistSet.load(_cfg["blacklist"]["path"])


@app.post("/search")
async def search(image: UploadFile, top_k_final: int = 20, text: str | None = None):
    raw = await image.read()
    query_image = Image.open(BytesIO(raw)).convert("RGB")
    query_vec = _encoder.encode([query_image])
    top_n = _cfg["retrieval"]["top_n_candidates"]

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_static = ex.submit(_static_searcher.search, query_vec, top_n)
        fut_dynamic = ex.submit(_dynamic_index.search, query_vec, top_n)
        s_scores, s_ids = fut_static.result()
        d_scores, d_ids = fut_dynamic.result()

    scores, ids = merge_candidates([s_scores[0], d_scores[0]], [s_ids[0], d_ids[0]])
    scores, ids = _blacklist.filter_out(ids, scores)
    scores, ids = text_rerank(scores, ids, text, metadata=None, alpha_visual=_cfg["rerank"]["alpha_visual"])
    scores, ids = top_k(scores, ids, top_k_final)

    return {"results": [{"product_id": int(pid), "score": float(s)} for s, pid in zip(scores, ids)]}
