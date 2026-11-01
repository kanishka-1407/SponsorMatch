"""
Baselines for startup -> investor ranking.

  1. Popularity : rank investors by number of training deals (ignores the startup)
  2. Affinity   : naive-Bayes style score  log P(inv) + sum_f w_f * log P(feature_f | inv)
                  Features: sector, stage, city, amount bin. Missing/Undisclosed features are skipped
                  (this is the cold-start behaviour the app will also use).

Run from repo root:   python src/baselines.py
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"

FEATURES = {"sector_group": "Unknown", "stage_bucket": "Undisclosed",
            "city_group": "Unknown", "amount_bin": "Undisclosed"}
KS = (1, 5, 10, 20)


# ------------------------------------------------------------------ models
class Popularity:
    name = "Popularity"

    def fit(self, train):
        self.counts = train["investor"].value_counts()
        self.investors = list(self.counts.index)
        return self

    def rank(self, row):
        return self.investors  # same list for everyone


class Affinity:
    name = "Affinity (rule-based)"

    def __init__(self, weights=None, alpha=0.5, prior_w=1.0):
        self.w = weights or {f: 1.0 for f in FEATURES}
        self.alpha = alpha
        self.prior_w = prior_w

    def fit(self, train):
        self.investors = list(train["investor"].value_counts().index)
        n = train["investor"].value_counts().reindex(self.investors).values.astype(float)
        self.log_prior = np.log(n / n.sum())
        self.tables = {}
        for f, missing in FEATURES.items():
            known = train[train[f] != missing]
            levels = sorted(known[f].unique())
            ct = pd.crosstab(known["investor"], known[f]).reindex(index=self.investors, columns=levels, fill_value=0)
            tot = ct.sum(axis=1).values[:, None]
            # smoothed P(level | investor); investors with no data for f fall back to uniform
            p = (ct.values + self.alpha) / (tot + self.alpha * len(levels))
            self.tables[f] = (levels, np.log(p))
        return self

    def scores(self, row):
        s = self.prior_w * self.log_prior.copy()
        for f, missing in FEATURES.items():
            v = row[f]
            if v == missing or pd.isna(v):
                continue
            levels, logp = self.tables[f]
            if v in levels:
                s += self.w[f] * logp[:, levels.index(v)]
        return s

    def rank(self, row):
        s = self.scores(row)
        return [self.investors[i] for i in np.argsort(-s, kind="stable")]


# ------------------------------------------------------------------ metrics
def evaluate(model, test, ks=KS):
    ranks = []
    top10_all = set()
    for _, row in test.iterrows():
        order = model.rank(row)
        try:
            r = order.index(row["investor"]) + 1
        except ValueError:
            r = np.inf  # investor unseen in training -> cannot be recovered
        ranks.append(r)
        top10_all.update(order[:10])
    ranks = np.array(ranks, dtype=float)
    out = {f"Hit@{k}": float((ranks <= k).mean()) for k in ks}
    out["MRR"] = float(np.where(np.isfinite(ranks), 1 / ranks, 0).mean())  # = MAP with 1 relevant item
    out["nDCG@10"] = float(np.where(ranks <= 10, 1 / np.log2(ranks + 1), 0).mean())
    out["Coverage@10"] = len(top10_all)  # distinct investors ever shown in a top-10
    out["n_test"] = len(test)
    return out


def tune(train, val):
    """Small grid search of feature weights on the validation split (never on test)."""
    best, best_score = None, -1
    grid = [0.0, 0.5, 1.0, 1.5]
    for ws, wst, wc, wa in itertools.product(grid, grid, grid, grid):
        for pw in (0.5, 1.0):
            m = Affinity({"sector_group": ws, "stage_bucket": wst, "city_group": wc, "amount_bin": wa},
                         alpha=0.5, prior_w=pw).fit(train)
            sc = evaluate(m, val)["nDCG@10"]
            if sc > best_score:
                best, best_score = (ws, wst, wc, wa, pw), sc
    ws, wst, wc, wa, pw = best
    return {"sector_group": ws, "stage_bucket": wst, "city_group": wc, "amount_bin": wa}, pw, best_score


def run_split(df, col, label):
    train, val, test = (df[df[col] == s] for s in ("train", "val", "test"))
    coverage = test.investor.isin(set(train.investor)).mean()
    w, pw, val_score = tune(train, val)
    models = [Popularity().fit(train),
              Affinity().fit(train),                                   # untuned: all weights 1
              Affinity(w, prior_w=pw).fit(train)]
    models[1].name = "Affinity (equal weights)"
    models[2].name = "Affinity (tuned on val)"
    rows = []
    for m in models:
        r = evaluate(m, test)
        r.update(split=label, model=m.name)
        rows.append(r)
    info = (f"{label}: train={len(train)}, val={len(val)}, test={len(test)}; "
            f"test investors seen in train: {coverage:.1%}; tuned weights={w}, prior_w={pw}, val nDCG@10={val_score:.3f}")
    return rows, info


def main():
    df = pd.read_csv(PROC / "deals_model.csv")
    all_rows, infos = [], []
    for col, label in [("split_random", "Random split (by company)"), ("split_time", "Time split (train<=2023, test 2025-26)")]:
        rows, info = run_split(df, col, label)
        all_rows += rows
        infos.append(info)
        print(info)
    res = pd.DataFrame(all_rows)[["split", "model", "Hit@1", "Hit@5", "Hit@10", "Hit@20", "MRR", "nDCG@10", "Coverage@10", "n_test"]]
    res.to_csv(REP / "baseline_results.csv", index=False)
    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    md = ["# Baseline results\n", "Task: given a startup's sector, stage, city and amount, rank the 298 modelled investors; "
          "the true lead investor of the round is the single relevant item.\n"]
    md += [f"- {i}" for i in infos]
    md += ["", res.round(3).to_markdown(index=False)]
    (REP / "baseline_results.md").write_text("\n".join(md), encoding="utf-8")


if __name__ == "__main__":
    main()
