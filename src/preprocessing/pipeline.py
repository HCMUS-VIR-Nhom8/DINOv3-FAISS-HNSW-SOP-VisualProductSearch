from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .image import (
    load_rgb, resize_keep_aspect, blur_score_laplacian,
    jpeg_artifact_score, letterbox, illumination_correction
)

@dataclass
class PreprocessOutput:
    image: object
    blur_score: float | None = None
    jpeg_score: float | None = None
    metadata: dict | None = None

class BaselinePreprocessor:
    def __init__(self, size=224):
        self.size = int(size)

    def __call__(self, path):
        img = load_rgb(path)
        return img.resize((self.size, self.size))

class ProposedPreprocessor:
    """
    Implements the currently reproducible part of:
    Orientation -> resize keep ratio -> quality flags -> letterbox -> CLAHE.

    Localization/segmentation hooks are intentionally optional because the
    attached method document describes Grounding-DINO/SAM as prototype
    components/checkpoint-dependent rather than guaranteed SOP labels.
    """
    def __init__(
        self,
        resize_long_side=1024,
        target_size=518,
        padding_ratio=0.10,
        blur_threshold=50.0,
        jpeg_threshold=8.0,
        use_illumination=True,
        clahe_clip_limit=2.0,
    ):
        self.resize_long_side = resize_long_side
        self.target_size = target_size
        self.padding_ratio = padding_ratio
        self.blur_threshold = blur_threshold
        self.jpeg_threshold = jpeg_threshold
        self.use_illumination = use_illumination
        self.clahe_clip_limit = clahe_clip_limit

    def __call__(self, path):
        img = load_rgb(path)
        img = resize_keep_aspect(img, self.resize_long_side)

        blur = blur_score_laplacian(img)
        jpeg = jpeg_artifact_score(img)

        # Quality flags are recorded, not used to silently delete images.
        img = letterbox(img, self.target_size)

        if self.use_illumination:
            img = illumination_correction(img, self.clahe_clip_limit)

        return PreprocessOutput(
            image=img,
            blur_score=blur,
            jpeg_score=jpeg,
            metadata={
                "blur_flag": blur < self.blur_threshold,
                "jpeg_flag": jpeg > self.jpeg_threshold,
            },
        )
