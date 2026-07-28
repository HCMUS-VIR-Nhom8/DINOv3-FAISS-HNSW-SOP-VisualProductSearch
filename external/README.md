# external/dinov3

Thu muc nay danh cho ma nguon clone tu ngoai - KHONG bat buoc phai dung neu
model.backend = "hf" trong configs/*.yaml (mac dinh, xem docs/method_dinov3.md).

## Neu can backend="torchhub" (clone repo DINOv3 local)

```bash
git submodule add https://github.com/facebookresearch/dinov3 external/dinov3
git submodule update --init --recursive
```

Sau do:
1. Xin quyen tai trong so tai: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   (dien form, Meta gui email chua link tai .pth cho tung bien the model)
2. Tai checkpoint bang `wget` (khong dung trinh duyet, theo khuyen cao cua repo goc):
   `bash scripts/download_dinov3_checkpoint.sh`
3. Trong configs/offline.yaml va configs/online.yaml, dat:
   ```yaml
   model:
     backend: torchhub
     torchhub_repo_dir: external/dinov3
     torchhub_arch: dinov3_vitb16   # hoac vits16, vitl16, convnext_base, ...
     torchhub_weights: /path/to/checkpoint.pth
   ```

Neu chi can embedding cap anh (image-level) cho retrieval, backend="hf" don gian
hon nhieu va khong can clone/submodule buoc nay.
