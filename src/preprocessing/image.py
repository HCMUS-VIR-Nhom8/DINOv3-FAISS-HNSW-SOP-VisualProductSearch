from __future__ import annotations
import cv2
import numpy as np
from PIL import Image, ImageOps

def load_rgb(path):
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).convert("RGB")

def resize_keep_aspect(img, target_long_side=1024):
    w, h = img.size
    scale = target_long_side / max(w, h)
    return img.resize(
        (max(1, round(w * scale)), max(1, round(h * scale))),
        Image.Resampling.LANCZOS,
    )

def blur_score_laplacian(img):
    arr = np.asarray(img)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def jpeg_artifact_score(img, block=8):
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY).astype(np.float32)
    vertical = np.abs(gray[:, 1:] - gray[:, :-1])
    horizontal = np.abs(gray[1:, :] - gray[:-1, :])
    vb = vertical[:, block - 1 :: block].mean() if vertical.shape[1] >= block else 0.0
    hb = horizontal[block - 1 :: block, :].mean() if horizontal.shape[0] >= block else 0.0
    return float((vb + hb) / 2.0)

def crop_with_padding(img, bbox, padding_ratio=0.10):
    x1, y1, x2, y2 = bbox
    w, h = img.size
    px, py = (x2 - x1) * padding_ratio, (y2 - y1) * padding_ratio
    box = (
        max(0, int(x1 - px)),
        max(0, int(y1 - py)),
        min(w, int(x2 + px)),
        min(h, int(y2 + py)),
    )
    return img.crop(box), box

def letterbox(img, size=518, fill=(114, 114, 114)):
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fill)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (left, top))
    return canvas

def illumination_correction(img, clip_limit=2.0):
    arr = np.asarray(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    out = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2RGB)
    return Image.fromarray(out)
