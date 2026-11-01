import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import baselines as B


def _toy():
    rows = []
    for _ in range(10):
        rows.append(dict(investor="A", sector_group="Fintech", stage_bucket="Seed", city_group="Mumbai", amount_bin="$1M-5M"))
    for _ in range(6):
        rows.append(dict(investor="B", sector_group="Edtech", stage_bucket="Series B", city_group="Pune", amount_bin="$20M-100M"))
    return pd.DataFrame(rows)


def test_affinity_prefers_matching_investor_over_popular_one():
    m = B.Affinity().fit(_toy())
    q = dict(sector_group="Edtech", stage_bucket="Series B", city_group="Pune", amount_bin="$20M-100M")
    assert m.rank(pd.Series(q))[0] == "B"          # B is less popular but fits


def test_missing_features_fall_back_to_popularity():
    m = B.Affinity().fit(_toy())
    q = dict(sector_group="Unknown", stage_bucket="Undisclosed", city_group="Unknown", amount_bin="Undisclosed")
    assert m.rank(pd.Series(q))[0] == "A"


def test_metrics_perfect_and_missing():
    train = _toy()
    m = B.Popularity().fit(train)
    test = pd.DataFrame([dict(investor="A"), dict(investor="Z")])   # Z never seen in training
    r = B.evaluate(m, test)
    assert r["Hit@1"] == 0.5 and abs(r["MRR"] - 0.5) < 1e-9
