import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.sop import load_sop

def proportional_sample(df, n, seed=42):
    rng = np.random.default_rng(seed)
    counts = df.groupby("class_id").size()
    eligible = counts[counts >= 2].copy()
    if eligible.sum() < n:
        raise ValueError(f"Only {eligible.sum()} eligible images; cannot sample {n}.")
    # Largest-remainder proportional allocation.
    raw = eligible / eligible.sum() * n
    alloc = np.floor(raw).astype(int)
    remainder = raw - alloc
    # Cap is the class size; allocate remaining samples by largest remainder.
    remaining = int(n - alloc.sum())
    for cls in remainder.sort_values(ascending=False).index:
        if remaining <= 0:
            break
        if alloc.loc[cls] < eligible.loc[cls]:
            alloc.loc[cls] += 1
            remaining -= 1
    # If caps prevented exact fill, distribute again.
    while remaining > 0:
        candidates = alloc[alloc < eligible]
        if candidates.empty:
            break
        for cls in candidates.index:
            if remaining <= 0:
                break
            alloc.loc[cls] += 1
            remaining -= 1

    parts = []
    for cls, k in alloc.items():
        if k <= 0:
            continue
        g = df[df.class_id == cls]
        idx = rng.choice(g.index.to_numpy(), size=int(k), replace=False)
        parts.append(g.loc[idx])
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    if len(out) != n:
        raise RuntimeError(f"Sampling produced {len(out)} rows instead of {n}.")
    return out

def make_query_gallery(df, query_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for cls, g in df.groupby("class_id"):
        g = g.sample(frac=1, random_state=seed + int(cls) % 100000)
        if len(g) < 2:
            continue
        nq = max(1, int(round(len(g) * query_fraction)))
        nq = min(nq, len(g)-1)
        q_idx = set(rng.choice(g.index.to_numpy(), size=nq, replace=False))
        split = np.where(g.index.isin(q_idx), "query", "gallery")
        gg = g.copy()
        gg["split"] = split
        rows.append(gg)
    return pd.concat(rows, ignore_index=True).sort_values("image_id").reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sop-root", required=True)
    ap.add_argument("--output", default="data/splits/sop_20k.csv")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--query-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_sop(args.sop_root)
    sampled = proportional_sample(df, args.n, args.seed)
    final = make_query_gallery(sampled, args.query_fraction, args.seed)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"Original: {len(df):,}")
    print(f"Sampled/evaluable: {len(final):,}")
    print(final["split"].value_counts().to_dict())
    print(f"Classes: {final.class_id.nunique():,}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
