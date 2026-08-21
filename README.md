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

### Option 1 - Run the Python scripts

This is the recommended way to reproduce the experiment from the repository source code:

```bash
python scripts/sample_sop.py --sop-root data/raw/Stanford_Online_Products

python scripts/run_baseline.py --config configs/baseline.yaml
python scripts/run_proposed.py --config configs/proposed.yaml

python scripts/evaluate.py --config configs/baseline.yaml
python scripts/evaluate.py --config configs/proposed.yaml

pytest -q
```

### Option 2 - Run the notebooks on Kaggle / Google Colab

The repository also provides experiment notebooks under `notebooks`/. These notebooks can be run on Kaggle or Google Colab, where GPU acceleration and notebook secrets are available.

Run the relevant notebooks in order and follow the paths/configuration specified by each notebook.

**DINOv3: Hugging Face access and token**

DINOv3 is a private/gated Hugging Face model. Before running the proposed DINOv3 pipeline:

1.Request access to the exact DINOv3 checkpoint used by this project on Hugging Face.
2. After access is granted, authenticate with Hugging Face.
3. On Kaggle/Colab, store the token in the platform's Secrets and expose it as HF_TOKEN.
4. Do not paste the token directly into notebook cells. Do not hard-code the token or commit it to Git.

For local/script execution, Hugging Face CLI authentication can be used:

```bash
hf auth login  --token SECRET_HF_TOKEN --add-to-git-credential
```

For Kaggle/Colab, the preferred approach is to read the token from the environment/secret manager:

```python
import os

hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise RuntimeError(
        "HF_TOKEN is not set. Add your Hugging Face token via Kaggle/Colab Secrets."
    )   
```

A valid token alone is not sufficient if access to the DINOv3 checkpoint has not been granted.

#### Reproducibility note

The script and notebook workflows are two execution interfaces for the same experiment, not two different methods. They should use the same dataset, split, model checkpoint, configuration, and evaluation protocol. The notebook workflow is especially useful for GPU execution and interactive inspection on Kaggle/Colab.
