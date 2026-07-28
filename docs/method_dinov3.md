# Phuong phap: DINOv3 cho trich xuat dac trung anh san pham

Repo goc: https://github.com/facebookresearch/dinov3

Backbone dung trong he thong: DINOv3 (Meta AI, self-supervised ViT/ConvNeXt).
Lay embedding cap anh (`pooler_output`) lam Global Embedding cho ca Offline
Indexing lan Online Retrieval - dam bao "offline dung checkpoint gi thi online
dung dung checkpoint do" (xem common/backbone/dinov3_encoder.py).

## Vi sao DINOv3 cho bai toan image-to-image retrieval

DINOv3 la self-supervised (khong can text pairing luc pretrain), dac trung hoc
duoc tap trung vao cau truc thi giac (hinh dang, texture, bo cuc) - phu hop
truy van hinh-anh-tim-hinh-anh hon la mo hinh alignment anh-text (CLIP/SigLIP).
Dac trung dense/patch-level chat luong cao con mo huong mo rong sau nay: crop
theo vung vat the, fine-grained matching theo tung phan anh.

## Tich hop: 2 backend, khong bat buoc phai clone repo goc

- `backend: hf` (mac dinh, xem configs/offline.yaml, configs/online.yaml) - dung
  Hugging Face `transformers.AutoModel`, khong can clone repo. Chi can accept
  license tren trang model + `huggingface-cli login`.
- `backend: torchhub` - clone repo vao `external/dinov3` (submodule), nap qua
  `torch.hub.load(..., source="local")`. Can khi dung tinh nang code goc chua
  duoc port sang HF (depth head, dino.txt, xuat dac trung patch-level day du).
  Xem external/README.md.

## Ap dung vao pipeline

- **Offline Indexing** (`offline/feature_extraction/extract_embeddings.py`):
  encode toan bo catalog 1 lan, luu embedding truoc khi build Static HNSW.
- **Online Retrieval** (`online/retrieval/ann_search.py` va cac pipeline trong
  `online/pipelines/`): encode anh query bang DUNG checkpoint da dung o offline.
