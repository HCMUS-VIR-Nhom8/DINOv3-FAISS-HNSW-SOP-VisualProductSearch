# Visual Product Search

Tim kiem san pham thuong mai bang hinh anh: Stanford Online Products + DINOv3
(feature extraction) + FAISS HNSW (indexing), chia lam 2 chu trinh doc lap
nhung dung chung backbone DINOv3 de dam bao "offline dung checkpoint gi thi
online dung nay":

- `offline/` : Indexing pipeline (batch, chay dinh ky / khi catalog thay doi)
- `online/`  : Retrieval pipeline + service (real-time, phuc vu query)
- `common/`  : Code dung chung giua 2 phia (backbone DINOv3, preprocessing, config, schema)
- `external/dinov3` : (tuy chon) submodule tro toi repo goc facebookresearch/dinov3

Xem `configs/` cho tham so pipeline, `docs/` cho rationale ky thuat, `scripts/`
cho lenh terminal tien ich, `notebooks/` cho demo end-to-end.

## Mapping voi so do kien truc

| So do | Thu muc / file tuong ung |
|---|---|
| Catalog Images → Preprocessing → DINOv3 Encoder → Static Main HNSW | `offline/feature_extraction/`, `offline/indexing/build_index.py`, `offline/pipelines/run_offline_indexing.py` |
| Query Image → Preprocessing → DINOv3 Encoder → Index HNSW → Top-N → Re-ranking → Top-K | `online/retrieval/ann_search.py`, `online/reranking/text_rerank.py`, `online/pipelines/run_online_retrieval.py` |
| Vector Query → ANN Parallel Search (Static HNSW + Dynamic Buffer) → Candidate Combination → Loai Blacklist → Text Re-ranking | `online/retrieval/dynamic_buffer.py`, `online/retrieval/blacklist.py`, `online/pipelines/run_online_retrieval_dynamic.py`, `online/api/app.py` |

Pipeline baseline va pipeline cai tien duoc **tach thanh 2 file rieng** (`run_online_retrieval.py` vs `run_online_retrieval_dynamic.py`) de de so sanh do tre / do chinh xac giua 2 kien truc khi viet bao cao.

## Cau truc thu muc day du

```
visual_product_search/
├── README.md
├── requirements.txt
├── configs/
│   ├── offline.yaml
│   └── online.yaml
├── common/
│   ├── config.py                # load_config(path)
│   ├── schema.py                 # CatalogItem, RetrievalResult
│   ├── backbone/
│   │   └── dinov3_encoder.py     # DINOv3Encoder (backend hf | torchhub) — dung chung
│   └── preprocessing/
│       └── image_io.py           # load_image() dung chung offline/online
├── external/
│   └── README.md                 # huong dan clone DINOv3 (backend=torchhub)
├── docs/
│   ├── method_dinov3.md
│   └── method_faiss_hnsw.md
├── notebooks/
│   └── 01_offline_indexing_and_online_demo.ipynb
├── offline/
│   ├── feature_extraction/
│   │   ├── sop_dataset.py        # SOPDataset (Stanford Online Products loader)
│   │   └── extract_embeddings.py
│   ├── indexing/
│   │   └── build_index.py        # build_hnsw_index() — Static Main HNSW
│   └── pipelines/
│       └── run_offline_indexing.py
├── online/
│   ├── retrieval/
│   │   ├── ann_search.py         # ANNSearcher — Static HNSW (baseline)
│   │   ├── dynamic_buffer.py     # DynamicBufferIndex — cai tien
│   │   └── blacklist.py          # BlacklistSet — cai tien
│   ├── reranking/
│   │   └── text_rerank.py        # merge_candidates, text_rerank, top_k
│   ├── pipelines/
│   │   ├── run_online_retrieval.py          # ONLINE baseline (Hinh 1.1)
│   │   ├── run_online_retrieval_dynamic.py  # ONLINE cai tien (Hinh 2.9)
│   │   └── manage_dynamic_catalog.py        # CLI add/remove san pham
│   └── api/
│       └── app.py                # FastAPI /search
├── scripts/
│   └── download_dinov3_checkpoint.sh
└── tests/
    ├── test_indexing.py
    └── test_preprocessing.py
```

## 1. Setup moi truong

```linux
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```windows
python -m venv .venv 
.venv\Scripts\activate
pip install -r requirements.txt
```

Khong can `pip install -e .` — cac package (`common`, `offline`, `online`) nam
ngay tai thu muc goc, chay bang `python -m` tu goc repo la du (xem muc 5-6).

`faiss-cpu` cai qua pip hoat dong tot tren Linux/macOS. Neu can GPU cho FAISS
hoac tren Windows, khuyen nghi cai qua conda: `conda install -c pytorch faiss-gpu`.

## 2. Chuan bi dataset Stanford Online Products

Tai tai: https://cvgl.stanford.edu/projects/lifted_struct/ (Stanford_Online_Products.zip)

Giai nen vao `data/raw/` sao cho co cau truc:
```
data/raw/Stanford_Online_Products/
├── Ebay_train.txt
├── Ebay_test.txt
└── <category>_final/*.jpg
```

## 3. Tich hop DINOv3 — 2 cach, khong bat buoc phai clone

Xem chi tiet ly do & huong dan tai `docs/method_dinov3.md` va `external/README.md`.
Mac dinh trong `configs/*.yaml`: `model.backend: hf` (Hugging Face Transformers,
khong can clone repo).

## 4. FAISS (HNSW)

`pip install faiss-cpu` la du (co san trong requirements.txt). Rationale lua
chon HNSW + Dynamic Buffer + Blacklist: `docs/method_faiss_hnsw.md`.

## 5. Chay pipeline BASELINE (Hinh 1.1)

```bash
# Offline: build Static Main HNSW tu catalog (Ebay_train.txt)
python -m offline.pipelines.run_offline_indexing --config configs/offline.yaml

# Online: tim kiem bang 1 anh query
python -m online.pipelines.run_online_retrieval \
    --image data/raw/Stanford_Online_Products/bicycle_final/some_image.jpg \
    --config configs/online.yaml
```

## 6. Chay pipeline CAI TIEN (Hinh 2.9)

```bash
# Them san pham moi ma khong rebuild HNSW
python -m online.pipelines.manage_dynamic_catalog add --image anh_moi.jpg --product-id 999001

# Go san pham khoi ket qua tra ve (blacklist)
python -m online.pipelines.manage_dynamic_catalog remove --product-id 12345

# Tim kiem: song song Static HNSW + Dynamic Buffer, loc blacklist, text re-rank
python -m online.pipelines.run_online_retrieval_dynamic \
    --image query.jpg --config configs/online.yaml --text "tui xach da"

# Hoac chay nhu mot service:
uvicorn online.api.app:app --reload --port 8000
```

Khi `dynamic.rebuild_threshold` (configs/online.yaml) bi vuot, chay lai
`offline.pipelines.run_offline_indexing` de gop buffer vao Static HNSW, sau do
goi `DynamicBufferIndex.reset()` va xoa `dynamic_buffer.index` cu.

## 7. Test nhanh (khong can model/dataset that)

```bash
pytest tests/ -v
```

## Ghi chu danh gia (goi y mo rong)

- Metric chuan cho SOP: Recall@K (K=1,10,100,1000) hoac mAP@K.
- So sanh truc tiep baseline (Hinh 1.1) vs cai tien (Hinh 2.9) tren cung 1 tap
  query de bao cao do tre (latency) va do chinh xac cho khoa luan/do an.
