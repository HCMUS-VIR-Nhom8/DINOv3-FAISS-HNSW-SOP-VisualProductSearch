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
    """Encoder wrapper for HuggingFace DINOv3 models.

    Parameters
    ----------
    model_name: str, optional
        Name of the HuggingFace model. Defaults to the tiny ConvNeXt variant
        used in the project.
    device: str, optional
        Device identifier ("cpu", "cuda", or "auto").
    """

    DEFAULT_MODEL = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

    def __init__(self, model_name: str = None, device: str = "auto"):
        self.device = _resolve_device(device)
        self.model_name = model_name or self.DEFAULT_MODEL
        # Processor handles image preprocessing (resize, normalization, etc.)
        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        # Load the HF model and move to the requested device
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device).eval()
        # Hidden size is stored in the config
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, images):
        """Encode a list of PIL images into L2‑normalized embeddings.
        Returns a NumPy array of shape (N, dim) with dtype float32.
        """
        # The processor returns a dict with a tensor under the key "pixel_values"
        inputs = self.processor(images, return_tensors="pt")
        batch = inputs["pixel_values"].to(self.device)
        # Model returns a dict; the CLS token is under "last_hidden_state"[:,0]
        outputs = self.model(batch)
        # CLS token (index 0) gives a representation per image
        cls_emb = outputs.last_hidden_state[:, 0]
        # L2‑normalize for cosine similarity usage
        return F.normalize(cls_emb, p=2, dim=1).cpu().numpy().astype("float32")


def _resolve_device(device: str) -> torch.device:
    """Utility to resolve "auto" to either cuda (if available) or cpu.
    This mirrors the logic used in previous notebooks.
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def get_encoder(encoder_type: str = "resnet", device: str = "auto", **kwargs):
    """Factory helper returning an encoder instance.

    Parameters
    ----------
    encoder_type: str
        Either "resnet" (baseline) or "dino" / "dinov3" (proposed).
    device: str
        Device identifier passed to the underlying encoder.
    **kwargs: dict
        Additional keyword arguments forwarded to the encoder constructor.
    """
    encoder_type = encoder_type.lower()
    if encoder_type in {"resnet", "baseline"}:
        return ResNet50Encoder(device=device, **kwargs)
    if encoder_type in {"dino", "dinov3", "proposed"}:
        return DINOv3Encoder(device=device, **kwargs)
    raise ValueError(f"Unsupported encoder_type: {encoder_type}")

# Export the public symbols expected by notebooks
__all__ = ["ResNet50Encoder", "DINOv3Encoder", "get_encoder"]
