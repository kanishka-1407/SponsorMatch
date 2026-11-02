import numpy as np
import pandas as pd
import torch
from model import InvestorRankingNN, NeuralRanker


def test_real_ranker_scores_are_ordered_and_finite():
    maps = {"sector_group": {"Fintech": 1}, "stage_bucket": {"Seed": 1},
            "city_group": {"Bengaluru": 1}, "amount_bin": {"$1M-5M": 1}}
    stats = {"amount_median": 6.0, "amount_mean": 6.0, "amount_std": 1.0,
             "age_mean": 3.0, "age_std": 1.0}
    torch.manual_seed(7)
    model = InvestorRankingNN(maps, 3, dropout=0)
    ranker = NeuralRanker(model, maps, stats, ["Fund A", "Fund B", "Fund C"], torch.device("cpu"))
    row = pd.Series({"sector_group": "Fintech", "stage_bucket": "Seed", "city_group": "Bengaluru",
                     "amount_bin": "$1M-5M", "amount_log10": 6.2, "company_age": 4})
    scores = ranker.scores(row)
    assert [name for name, _ in scores] == ranker.rank(row)
    assert len(scores) == 3
    assert all(np.isfinite(score) and 0 <= score <= 1 for _, score in scores)
    assert [score for _, score in scores] == sorted((score for _, score in scores), reverse=True)
