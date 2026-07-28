# Phuong phap: FAISS HNSW + Dynamic Buffer + Blacklist

Repo goc: https://github.com/facebookresearch/faiss (chi can `pip install faiss-cpu`,
khong can clone/build tu source tru khi can custom kernel).

## Vi sao HNSW cho quy mo hien tai

`IndexHNSWFlat` khong can buoc train() rieng, ho tro `add()` gia tang, phu hop
catalog thuong mai lien tuc them san pham moi va quy mo duoi ~1-2 trieu anh
(Stanford Online Products co ~120k anh). Neu catalog vuot vai chuc trieu anh,
can chuyen sang ho IVF+PQ (huong mo rong, chua trien khai trong baseline nay).

## Van de: HNSW khong ho tro xoa/them hieu qua theo thoi gian thuc

Rebuild toan bo HNSW moi lan co san pham moi hoac bi go la qua ton kem cho he
thong real-time. Giai phap ap dung, dung theo Hinh 2.9:

- **Dynamic Buffer Index** (`online/retrieval/dynamic_buffer.py`) - mot Flat
  index rieng, brute-force, chi chua san pham MOI them sau lan build Static
  HNSW gan nhat. Buffer nho nen search vet can van du nhanh; khi buffer vuot
  `rebuild_threshold` (configs/online.yaml), gop lai vao Static HNSW roi reset.
- **Blacklist Set** (`online/retrieval/blacklist.py`) - Roaring Bitmap chua id
  san pham bi go/an, loc o buoc hau-xu-ly thay vi xoa that khoi HNSW.
- **ANN Parallel Search + Candidate Combination**
  (`online/pipelines/run_online_retrieval_dynamic.py`) - search song song ca 2
  index, gop ung vien giu diem cao nhat khi trung id, loc blacklist, roi Text
  Re-ranking.

## Tach biet baseline (Hinh 1.1) vs cai tien (Hinh 2.9)

`run_online_retrieval.py` chi dung Static HNSW (dung Hinh 1.1). 
`run_online_retrieval_dynamic.py` la ban day du theo Hinh 2.9. Giu 2 file rieng
de de so sanh do tre / do chinh xac giua 2 kien truc trong bao cao/khoa luan.
