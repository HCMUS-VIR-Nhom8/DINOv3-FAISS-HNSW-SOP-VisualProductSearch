"""
Loader cho Stanford Online Products (SOP) dataset.
Tai dataset tai: https://cvgl.stanford.edu/projects/lifted_struct/

Cau truc file Ebay_train.txt / Ebay_test.txt (cach nhau boi khoang trang, co header):
    image_id class_id super_class_id path
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from PIL import Image
from torch.utils.data import Dataset


@dataclass
class SOPItem:
    image_id: int
    class_id: int
    super_class_id: int
    path: Path


def _parse_list_file(root: Path, list_file: str) -> List[SOPItem]:
    items = []
    with open(root / list_file, "r") as f:
        next(f)  # bo header
        for line in f:
            image_id, class_id, super_class_id, rel_path = line.strip().split()
            items.append(
                SOPItem(
                    image_id=int(image_id),
                    class_id=int(class_id),
                    super_class_id=int(super_class_id),
                    path=root / rel_path,
                )
            )
    return items


class SOPDataset(Dataset):
    """Dung cho ca 2 vai tro: 'catalog' (offline indexing, dung Ebay_train.txt)
    va 'query' (mo phong truy van / danh gia, dung Ebay_test.txt).
    Tra ve anh PIL goc (chua transform) vi DINOv3Encoder tu xu ly preprocessing ben trong."""

    def __init__(self, root: str, list_file: str, transform: Optional[callable] = None):
        self.root = Path(root)
        self.items = _parse_list_file(self.root, list_file)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = Image.open(item.path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, item
