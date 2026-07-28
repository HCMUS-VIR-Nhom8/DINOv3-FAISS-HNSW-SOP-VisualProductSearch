from PIL import Image

from common.preprocessing.image_io import load_image


def test_load_image_convert_rgb(tmp_path):
    p = tmp_path / "test.png"
    Image.new("L", (50, 50)).save(p)  # anh grayscale
    img = load_image(str(p))
    assert img.mode == "RGB"
    assert img.size == (50, 50)
