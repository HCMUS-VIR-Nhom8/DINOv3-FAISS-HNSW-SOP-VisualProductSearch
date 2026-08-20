from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = ["image_id", "class_id", "super_class_id", "path"]

def read_sop_txt(path: str | Path, source_split: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    if source_split is not None:
        df["source_split"] = source_split
    return df

def load_sop(root: str | Path) -> pd.DataFrame:
    root = Path(root)
    train = read_sop_txt(root / "Ebay_train.txt", "train")
    test = read_sop_txt(root / "Ebay_test.txt", "test")
    return pd.concat([train, test], ignore_index=True)

def attach_split(df: pd.DataFrame, split_csv: str | Path) -> pd.DataFrame:
    split = pd.read_csv(split_csv)
    needed = {"image_id", "split"}
    if not needed.issubset(split.columns):
        raise ValueError(f"CSV must contain {needed}")
    out = df.merge(split[["image_id", "split"]], on="image_id", how="inner")
    if out.empty:
        raise ValueError("No image_id overlap between SOP metadata and split CSV.")
    return out

def resolve_image_path(image_root: str | Path, relative_path: str) -> Path:
    return Path(image_root) / relative_path
