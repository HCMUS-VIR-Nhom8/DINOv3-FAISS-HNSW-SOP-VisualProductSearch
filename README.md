# Visual Product Search - SOP 20K Experiments

This repository is intentionally organized around the thesis experiment:

1. Stratified sampling: SOP ~120K -> target 20K.
2. Baseline: image -> resize -> pretrained ResNet50 -> global average pooling -> L2 normalization -> exact cosine search.
3. Proposed: preprocessing -> pretrained DINOv3 [CLS] embedding -> L2 normalization -> FAISS HNSW -> Top-N -> metadata re-ranking.
4. Evaluation: Recall@K, mAP, latency, memory, qualitative retrieval.

## Important evaluation protocol

The sampled set is split **within product identity (`class_id`)** into query/gallery so every evaluated query has at least one positive gallery image. This is a retrieval protocol, not a supervised training split. The split is created after the 20K stratified sample, so both query and gallery contain images from the same product identities by design.

The source document states that SOP is sampled by `class-id`, that the resulting subset is used as gallery/query, and that Recall@K, mAP, latency and memory are the target evaluation metrics. The preprocessing sequence is Orientation -> Resize -> Quality Check -> Localization -> Segmentation -> Crop+Padding -> Letterbox -> Illumination Correction.

### Metadata caveat

SOP's `Ebay_*.txt` provides `image_id`, `class_id`, `super_class_id`, and image path. It does **not** provide title/brand/description text. Therefore this implementation's metadata re-ranking is a controlled `super_class_id` candidate-consistency heuristic, not text/brand re-ranking. Do not report it as textual metadata.

## Setup

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Put SOP under:

`data/raw/Stanford_Online_Products/`

with:

- `Ebay_train.txt`
- `Ebay_test.txt`
- category image folders

DINOv3 is a gated Hugging Face model; request access/login before running the proposed pipeline.

## Run

```bash
python scripts/sample_sop.py --sop-root data/raw/Stanford_Online_Products

python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_proposed.py --config configs/proposed.yaml

python scripts/evaluate.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/proposed.yaml

pytest -q
```

## Recommended experiment table

| Method | Encoder | Retrieval | R@1 | R@10 | R@100 | mAP | Latency/query | Memory |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | ResNet50 | Exact cosine | | | | | | |
| Proposed-no-rerank | DINOv3 | HNSW | | | | | | |
| Proposed | DINOv3 | HNSW + metadata | | | | | | |

The middle row is important: it isolates the effect of HNSW from the effect of re-ranking.

## Memory definition used in the experiment

Report at least two memory views:

- `gallery_embedding_memory_mb`: raw gallery embedding storage.
- `hnsw_index_storage_mb`: serialized HNSW index size for the proposed method.
- `total_retrieval_storage_mb`: storage cost of the retrieval representation/index.
- `rss_delta_after_gallery_encoding_mb`: process RSS change as an engineering/runtime indicator.

This avoids confusing model memory with index/embedding storage.
