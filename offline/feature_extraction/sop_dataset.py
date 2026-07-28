"""Loader cho Stanford Online Products (SOP) - dung lam catalog cho Offline Indexing.
Tai dataset tai: https://cvgl.stanford.edu/projects/lifted_struct/

Cau truc file Ebay_train.txt / Ebay_test.txt (cach nhau boi khoang trang, co header):
    image_id class_id super_class_id path
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from torch.utils.data import Dataset

from common.preprocessing.image_io import load_image
from common.schema import CatalogItem


def _parse_list_file(root: Path, list_file: str) -> List[CatalogItem]:
    items = []
    with open(root / list_file, "r") as f:
        next(f)  # bo header
        for line in f:
            image_id, class_id, super_class_id, rel_path = line.strip().split()
            items.append(
                CatalogItem(
                    product_id=int(image_id),
                    image_path=str(root / rel_path),
                    class_id=int(class_id),
                    super_class_id=int(super_class_id),
                )
            )
    return items


class SOPDataset(Dataset):
    def __init__(self, root: str, list_file: str):
        self.root = Path(root)
        self.items = _parse_list_file(self.root, list_file)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        image = load_image(item.image_path)
        return image, item
