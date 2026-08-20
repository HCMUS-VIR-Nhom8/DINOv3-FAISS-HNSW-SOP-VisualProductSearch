from __future__ import annotations
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
from transformers import AutoImageProcessor, AutoModel

class ResNet50Encoder:
    def __init__(self, device="auto", pretrained=True):
        self.device = _resolve_device(device)
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        self.transform = weights.transforms() if weights else ResNet50_Weights.DEFAULT.transforms()
        self.model = torch.nn.Sequential(*list(model.children())[:-1]).to(self.device).eval()
        self.dim = 2048

    @torch.inference_mode()
    def encode(self, images):
        batch = torch.stack([self.transform(im) for im in images]).to(self.device)
        z = self.model(batch).flatten(1)
        return F.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")

class DINOv3Encoder:
    def __init__(self, model_name, device="auto"):
        self.device = _resolve_device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # For ViT DINOv3, last_hidden_state[:, 0] is the class/global token.
        z = outputs.last_hidden_state[:, 0]
        return F.normalize(z, p=2, dim=1).cpu().numpy().astype("float32")

def _resolve_device(device):
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device
