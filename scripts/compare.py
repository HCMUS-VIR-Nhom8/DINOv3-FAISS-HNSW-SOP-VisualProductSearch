import argparse, json
from pathlib import Path
import pandas as pd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--baseline", default="outputs/baseline/metrics.json")
    ap.add_argument("--proposed", default="outputs/proposed/metrics.json")
    ap.add_argument("--output", default="outputs/comparison/metrics_comparison.csv")
    args=ap.parse_args()
    rows=[]
    for name,path in [("baseline",args.baseline),("proposed",args.proposed)]:
        m=json.load(open(path,encoding="utf-8")); m["method"]=name; rows.append(m)
    df=pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(args.output,index=False)
    print(df.to_string(index=False))

if __name__=="__main__":
    main()
