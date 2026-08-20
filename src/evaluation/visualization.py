from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_metric_comparison(comparison_csv, output_path):
    df = pd.read_csv(comparison_csv)
    metrics = [c for c in df.columns if c.lower().startswith("recall@")] + ["mAP"]
    ax = df.set_index("method")[metrics].T.plot(kind="bar", figsize=(12, 6))
    ax.set_ylabel("Score")
    ax.set_title("Retrieval quality comparison")
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
