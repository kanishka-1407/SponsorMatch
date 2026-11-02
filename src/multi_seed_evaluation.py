"""Compare neural and hybrid rankings over five random seeds."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from baselines import Affinity, evaluate
from model import set_seed, train_model, NeuralRanker


def main():
    df = pd.read_csv(ROOT / "data/processed/deals_model.csv")
    train, val, test = [df[df.split_random == split].copy() for split in ("train", "val", "test")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = []
    for seed in (11, 23, 42, 57, 91):
        set_seed(seed)
        model, maps, stats, investors, best = train_model(train, val, device)
        neural = NeuralRanker(model, maps, stats, investors, device)
        affinity = Affinity().fit(train)
        class Hybrid:
            def rank(self, row):
                a, n = affinity.rank(row), neural.rank(row)
                ar, nr = {x:i for i,x in enumerate(a)}, {x:i for i,x in enumerate(n)}
                return sorted(investors, key=lambda x: (ar.get(x, len(investors)) + nr.get(x, len(investors)), x.casefold()))
        for label, ranker in (("neural", neural), ("hybrid", Hybrid())):
            result = evaluate(ranker, test)
            result.update(seed=seed, model=label, best_val_nDCG10=best)
            output.append(result)
    result = pd.DataFrame(output)
    result.to_csv(ROOT / "reports/multi_seed_results.csv", index=False)
    summary = result.groupby("model")[ ["Hit@1", "Hit@5", "Hit@10", "MRR", "nDCG@10"] ].agg(["mean", "std"])
    summary.to_csv(ROOT / "reports/multi_seed_summary.csv")
    print(summary.round(4).to_string())

if __name__ == "__main__":
    main()
