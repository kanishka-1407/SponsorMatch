"""Run with:  pytest -q   (from repo root)"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import preprocess as P
import mappings as M

PROC = Path(__file__).resolve().parents[1] / "data" / "processed"


def test_normalise_key_merges_variants():
    assert P.normalise_key("Sequoia Capital India") == P.normalise_key("Sequoia Capital")
    assert P.normalise_key("Accel India") == P.normalise_key("Accel")
    assert P.normalise_key("Bill & Melinda Gates Foundation") == P.normalise_key("Bill and Melinda Gates Foundation")


def test_junk_detection():
    assert P.is_junk("few Private Investors")
    assert P.is_junk("Godrej Family Office")
    assert not P.is_junk("Blume Ventures")
    assert not P.is_junk("Indian Angel Network")


def test_every_sector_and_stage_mapped():
    raw = pd.read_csv(PROC.parent / "raw" / "india_startup_funding_sample.csv")
    assert raw.sector.dropna().isin(M.SECTOR_LOOKUP).all()
    assert raw.stage.isin(M.STAGE_MAP).all()


def test_no_company_leaks_across_split():
    d = pd.read_csv(PROC / "deals_model.csv")
    assert d.groupby("company_id").split_random.nunique().max() == 1


def test_model_set_has_only_known_investors_with_min_deals():
    d = pd.read_csv(PROC / "deals_model.csv")
    assert d.investor.notna().all()
    assert (d.investor_deal_count >= M.MIN_DEALS).all()


def test_profiles_match_model_data():
    p = pd.read_csv(PROC / "investor_profiles.csv")
    assert p.investor.is_unique and (p.n_deals >= M.MIN_DEALS).all()


def test_pool_one_row_per_company():
    pool = pd.read_csv(PROC / "startups_pool.csv")
    assert pool.company_id.is_unique
