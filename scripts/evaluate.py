import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from src.evaluation.metrics import evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out = Path(cfg["output"]["dir"])
    df = pd.read_csv(cfg["data"]["csv"])
    q = df[df.split == cfg["data"]["query_split"]].reset_index(drop=True)
    g = df[df.split == cfg["data"]["gallery_split"]].reset_index(drop=True)
    z = np.load(out / cfg["output"]["results"])
    m = evaluate(z["ids"], q.class_id.values, g.class_id.values, cfg["evaluation"]["ks"])
    print(json.dumps(m, indent=2))
    (out / "metrics_recomputed.json").write_text(json.dumps(m, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
