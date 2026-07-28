# third_party/

Thu muc nay danh cho ma nguon clone tu ngoai (KHONG bat buoc phai dung neu chon backend="hf").

## Neu muon dung backend="torchhub" (clone repo DINOv3 local)

```bash
git submodule add https://github.com/facebookresearch/dinov3 third_party/dinov3
```

Sau do:
1. Xin quyen tai trong so tai: https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/
   (dien form, Meta se gui email chua link tai .pth cho tung bien the model)
2. Tai checkpoint bang `wget` (KHONG dung trinh duyet, theo khuyen cao cua repo goc)
3. Trong `configs/config.yaml`, dat:
   ```yaml
   encoder:
     backend: torchhub
     torchhub_repo_dir: third_party/dinov3
     torchhub_arch: dinov3_vitb16       # hoac vits16, vitl16, convnext_base, ...
     torchhub_weights: /path/to/checkpoint.pth
   ```

Neu chi can embedding cap anh (image-level) cho retrieval nhu baseline nay, backend="hf"
(Hugging Face Transformers) don gian hon nhieu va KHONG can clone/submodule buoc nay -
xem README.md o thu muc goc.
