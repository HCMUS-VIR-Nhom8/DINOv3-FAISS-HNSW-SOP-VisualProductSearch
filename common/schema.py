"""Dataclasses dung chung: Catalog item, Retrieval result."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class CatalogItem:
    """1 san pham trong catalog. Voi dataset Stanford Online Products, product_id
    tuong ung image_id, con class_id/super_class_id la nhan phan cap muc do tuong dong."""
    product_id: int
    image_path: str
    class_id: Optional[int] = None
    super_class_id: Optional[int] = None
    title: Optional[str] = None  # dung cho Text Re-ranking khi co metadata


@dataclass
class RetrievalResult:
    product_id: int
    visual_score: float
    text_score: float = 0.0
    final_score: float = 0.0
