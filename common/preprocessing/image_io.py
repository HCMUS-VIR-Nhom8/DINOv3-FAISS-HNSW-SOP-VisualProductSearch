"""Doc anh dung chung cho ca Offline Indexing va Online Retrieval - dam bao 2 phia
xu ly anh dau vao giong het nhau (convert RGB) truoc khi dua vao DINOv3 Encoder.
Resize/normalize cu the do processor cua tung backend (HF hoac torchhub) dam nhiem
ben trong common/backbone/dinov3_encoder.py, de tranh lech tien xu ly offline/online."""
from PIL import Image


def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")
