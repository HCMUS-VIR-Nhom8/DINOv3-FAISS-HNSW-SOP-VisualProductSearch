"""
Boc (wrap) DINOv3 de trich xuat dac trung anh (feature extraction).
Ho tro 2 cach nap model - khong bat buoc phai clone repo:

  - backend="hf"       : dung Hugging Face `transformers` (KHUYEN NGHI cho baseline).
                          Chi can accept license cua model tren trang Hugging Face
                          + `huggingface-cli login`, khong can clone code goc.
  - backend="torchhub" : clone repo facebookresearch/dinov3 ve local roi nap qua
                          torch.hub.load(..., source="local"). Dung khi can cac bien
                          the/dau ra ma HF chua ho tro (vd depth head, dino.txt, ...).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image


class DINOv3Encoder:
    def __init__(
        self,
        backend: str = "hf",
        model_name: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        torchhub_repo_dir: Optional[str] = None,
        torchhub_arch: Optional[str] = None,
        torchhub_weights: Optional[str] = None,
        device: str = "cuda",
        image_size: int = 224,
    ):
        self.backend = backend
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size

        if backend == "hf":
            self._init_hf(model_name)
        elif backend == "torchhub":
            self._init_torchhub(torchhub_repo_dir, torchhub_arch, torchhub_weights)
        else:
            raise ValueError(f"Backend khong ho tro: {backend}")

        self.model.eval().to(self.device)

    # ---------- Backend: Hugging Face Transformers ----------
    def _init_hf(self, model_name: str):
        from transformers import AutoImageProcessor, AutoModel

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._forward = self._forward_hf

    def _forward_hf(self, images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs = self.model(**inputs)
        # pooler_output: [B, D] - embedding cap anh (image-level), dung cho retrieval
        return outputs.pooler_output

    # ---------- Backend: torch.hub tu repo clone local ----------
    def _init_torchhub(self, repo_dir, arch, weights):
        if repo_dir is None or arch is None:
            raise ValueError("Can torchhub_repo_dir va torchhub_arch khi backend='torchhub'")
        repo_dir = str(Path(repo_dir).resolve())
        sys.path.append(repo_dir)  # cho phep import them package `dinov3` neu can

        from torchvision.transforms import v2

        self.model = torch.hub.load(repo_dir, arch, source="local", weights=weights)
        self._transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((self.image_size, self.image_size), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self._forward = self._forward_torchhub

    def _forward_torchhub(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self._transform(img) for img in images]).to(self.device)
        with torch.inference_mode():
            feats = self.model(batch)  # backbone ViT tra ve CLS-pooled feature [B, D]
        return feats

    # ---------- API dung chung ----------
    @torch.no_grad()
    def encode(self, images: List[Image.Image], normalize: bool = True) -> np.ndarray:
        """Tra ve embedding [N, D] dang numpy float32, mac dinh da L2-normalize
        de search bang inner-product tuong duong cosine similarity."""
        feats = self._forward(images)
        if normalize:
            feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats.cpu().numpy().astype("float32")
