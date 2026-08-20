from PIL import Image
from src.preprocessing.image import resize_keep_aspect, letterbox

def test_resize_keep_aspect():
    im = Image.new("RGB", (400, 200))
    out = resize_keep_aspect(im, 100)
    assert out.size == (100, 50)

def test_letterbox():
    im = Image.new("RGB", (400, 200))
    out = letterbox(im, 224)
    assert out.size == (224, 224)
