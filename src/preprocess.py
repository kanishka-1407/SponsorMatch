"""
Preprocessing pipeline: raw funding rounds -> clean, model-ready tables.

Run from the repo root:   python src/preprocess.py
Outputs go to data/processed/ and reports/.
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import mappings as M

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "india_startup_funding_sample.csv"
OUT = ROOT / "data" / "processed"
REP = ROOT / "reports"
LOG = []  # lines for the data-quality report


def log(msg):
    print(msg)
    LOG.append(msg)


# ------------------------------------------------------------------ 1. load + audit
def load_and_audit():
    df = pd.read_csv(RAW)
    log("## 1. Raw audit")
    log(f"- Rows: {len(df)}, columns: {df.shape[1]}")
    log(f"- Duplicate round_id: {df.round_id.duplicated().sum()}; exact duplicate rows: {df.duplicated().sum()}")
    nulls = df.isna().sum()
    log("- Missing values: " + ", ".join(f"{c}={n}" for c, n in nulls.items() if n))
    log(f"- Date range: {df.announced_date.min()} to {df.announced_date.max()}")
    return df


# ------------------------------------------------------------------ 2. basic cleaning
def clean_text(s):
    if pd.isna(s):
        return np.nan
    s = unicodedata.normalize("NFKC", str(s)).replace("\u2019", "'")
    s = re.sub(r"\s+", " ", s).strip()
    return s or np.nan


def basic_clean(df):
    log("\n## 2. Basic cleaning")
    for c in ["company_name", "company_domain", "city", "sector", "stage", "lead_investor", "news_excerpt"]:
        df[c] = df[c].map(clean_text)
    df["announced_date"] = pd.to_datetime(df["announced_date"], errors="coerce")
    bad = df.announced_date.isna().sum()
    mism = (df.announced_date.dt.year != df.year).sum()
    log(f"- Unparseable dates: {bad}; year column disagrees with date: {mism}")
    df["company_domain"] = df["company_domain"].str.lower().str.replace(r"^www\.", "", regex=True)
    df = df.drop_duplicates("round_id").reset_index(drop=True)
    # company identity = domain (names can repeat / vary; domains are unique per company)
    df["company_id"] = pd.factorize(df["company_domain"])[0]
    log(f"- Unique companies (by domain): {df.company_id.nunique()}; by name: {df.company_name.str.lower().nunique()}")
    return df


# ------------------------------------------------------------------ 3. sector / stage / city
def standardise_categoricals(df):
    log("\n## 3. Sector, stage, city")
    df["sector_raw"] = df["sector"]
    df["sector_group"] = df["sector"].map(M.SECTOR_LOOKUP)
    unmapped = df.loc[df.sector.notna() & df.sector_group.isna(), "sector"].unique()
    assert len(unmapped) == 0, f"Unmapped sectors: {unmapped}"
    df["sector_group"] = df["sector_group"].fillna("Unknown")
    log(f"- Sector: {df.sector_raw.nunique()} raw labels -> {df.sector_group.nunique()-1} groups; unknown rows: {(df.sector_group=='Unknown').sum()}")

    df["stage_raw"] = df["stage"]
    df["stage_bucket"] = df["stage"].map(M.STAGE_MAP)
    unmapped = df.loc[df.stage.notna() & df.stage_bucket.isna(), "stage"].unique()
    assert len(unmapped) == 0, f"Unmapped stages: {unmapped}"
    df["stage_bucket"] = df["stage_bucket"].fillna("Undisclosed")
    log(f"- Stage: {df.stage_raw.nunique()} raw labels -> {df.stage_bucket.nunique()} buckets")

    def fix_city(c):
        if pd.isna(c):
            return np.nan
        key = c.lower()
        if key in M.CITY_ALIASES:
            return M.CITY_ALIASES[key]
        return c.title() if c.islower() else c

    df["city_raw"] = df["city"]
    df["city_clean"] = df["city"].map(fix_city)

    def city_group(c):
        if pd.isna(c):
            return "Unknown"
        if c in M.NCR:
            return "Delhi NCR"
        if c in M.HUBS:
            return c
        if c in M.INTERNATIONAL:
            return "International"
        return "Other India"

    df["city_group"] = df["city_clean"].map(city_group)
    log(f"- City: {df.city_raw.nunique()} raw spellings -> {df.city_clean.nunique()} cities -> {df.city_group.nunique()} groups")
    return df


# ------------------------------------------------------------------ 4. amount + age
def numeric_features(df):
    log("\n## 4. Amount and company age")
    df["amount_known"] = df["amount_usd"].notna() & ~df["amount_status"].isin(["not_disclosed", "failed_check"])
    df.loc[~df.amount_known, "amount_usd"] = np.nan
    df["amount_unverified"] = df["amount_status"].eq("unchecked")
    df["amount_suspect_low"] = df["amount_usd"] < 5_000
    df["amount_log10"] = np.log10(df["amount_usd"])
    df["amount_bin"] = pd.cut(df["amount_usd"], M.AMOUNT_BINS, labels=M.AMOUNT_LABELS, right=False).astype("object")
    df["amount_bin"] = df["amount_bin"].fillna("Undisclosed")
    log(f"- Amount known: {df.amount_known.sum()} / {len(df)}; suspiciously low (<$5K): {df.amount_suspect_low.sum()}; unverified: {df.amount_unverified.sum()}")
    log("- Outliers are kept (a $1B round is real); models should use log10 amount.")

    df["company_age"] = df["year"] - df["founded_year"]
    df["founded_year_clipped"] = df["founded_year"].eq(df.founded_year.min())
    df["age_bin"] = pd.cut(df["company_age"], M.AGE_BINS, labels=M.AGE_LABELS).astype("object")
    log(f"- Negative ages: {(df.company_age<0).sum()}; founded_year == {df.founded_year.min()} (likely clipped at lower bound): {df.founded_year_clipped.sum()} rows")
    return df


# ------------------------------------------------------------------ 5. investors
def normalise_key(name):
    s = unicodedata.normalize("NFKC", name).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [t for t in s.split() if t not in M.KEY_STOPWORDS]
    return " ".join(toks) if toks else s.strip()


def is_junk(name):
    low = name.lower()
    return any(re.search(p, low) for p in M.JUNK_INVESTOR_PATTERNS)


def investor_type(name):
    low = name.lower()
    if "angel" in low:
        return "Angel network/fund"
    if any(k in low for k in M.INSTITUTION_KEYWORDS):
        return "Institution/Govt/Incubator"
    return "VC/PE/Other"


def clean_investors(df):
    log("\n## 5. Investors")
    raw = df["lead_investor"]
    log(f"- Blank lead_investor: {raw.isna().sum()} rows")
    junk_mask = raw.notna() & raw.map(lambda x: isinstance(x, str) and is_junk(x))
    log(f"- Junk / non-identifiable entries removed: {junk_mask.sum()} rows ({raw[junk_mask].nunique()} distinct)")
    valid = raw.where(~junk_mask)

    names = valid.dropna()
    keys = names.map(normalise_key)
    # manual aliases (applied on the key)
    alias_keys = {normalise_key(k): v for k, v in M.MANUAL_INVESTOR_ALIASES.items()}
    mapped_key = keys.map(lambda k: normalise_key(alias_keys[k]) if k in alias_keys else k)

    # canonical name = most frequent original spelling within the key group
    canon = {}
    for k, grp in names.groupby(mapped_key):
        canon[k] = Counter(grp).most_common(1)[0][0]
    for k, v in alias_keys.items():
        canon[normalise_key(v)] = v
    investor_canon = mapped_key.map(canon)

    df["investor_raw"] = raw
    df["investor_is_junk"] = junk_mask
    df["investor"] = investor_canon.reindex(df.index)
    log(f"- Distinct investor spellings: {names.nunique()} -> canonical investors: {df.investor.nunique()}")

    # audit tables ------------------------------------------------------------
    alias = (pd.DataFrame({"raw": names, "canonical": investor_canon})
             .groupby(["canonical", "raw"]).size().rename("n_rows").reset_index())
    merged = alias.groupby("canonical").filter(lambda g: len(g) > 1)
    merged.to_csv(OUT / "investor_alias_map.csv", index=False)
    log(f"- Alias map saved: {merged.canonical.nunique()} canonical names absorbed >1 spelling")

    # near-duplicate candidates for MANUAL review (not auto-merged)
    try:
        from rapidfuzz import fuzz, process
        counts = df.investor.value_counts()
        pool = counts[counts >= 2].index.tolist()
        rows = []
        for i, a in enumerate(pool):
            for b, score, _ in process.extract(a, pool[i + 1:], scorer=fuzz.token_sort_ratio, score_cutoff=88, limit=3):
                rows.append((a, b, int(score), int(counts[a]), int(counts[b])))
        pd.DataFrame(rows, columns=["name_a", "name_b", "similarity", "n_a", "n_b"]) \
            .sort_values("similarity", ascending=False) \
            .to_csv(REP / "investor_near_duplicates_for_review.csv", index=False)
        log(f"- Near-duplicate investor pairs flagged for manual review: {len(rows)} (reports/investor_near_duplicates_for_review.csv)")
    except ImportError:
        log("- rapidfuzz not installed: near-duplicate review skipped")

    df["investor_type"] = df["investor"].map(lambda x: investor_type(x) if isinstance(x, str) else np.nan)
    df["has_investor"] = df["investor"].notna()
    return df


# ------------------------------------------------------------------ 6. modelling flags + splits
def modelling_flags(df):
    log("\n## 6. Modelling eligibility and splits")
    counts = df["investor"].value_counts()
    keep = set(counts[counts >= M.MIN_DEALS].index)
    df["investor_deal_count"] = df["investor"].map(counts)
    df["investor_modelled"] = df["investor"].isin(keep)
    df["sector_known"] = df["sector_group"].ne("Unknown")
    df["model_eligible"] = df["investor_modelled"] & df["sector_known"]
    log(f"- Investors with >= {M.MIN_DEALS} deals: {len(keep)}; eligible rows: {df.model_eligible.sum()} "
        f"({df.loc[df.model_eligible,'company_id'].nunique()} companies)")

    # split BY COMPANY so the same startup never appears in train and test
    rng = np.random.default_rng(M.RANDOM_SEED)
    elig_companies = df.loc[df.model_eligible, "company_id"].unique()
    rng.shuffle(elig_companies)
    n = len(elig_companies)
    assign = {}
    for i, c in enumerate(elig_companies):
        assign[c] = "train" if i < 0.70 * n else ("val" if i < 0.85 * n else "test")
    df["split_random"] = df["company_id"].map(assign)
    df.loc[~df.model_eligible, "split_random"] = np.nan

    df["split_time"] = np.where(df.year <= 2023, "train", np.where(df.year == 2024, "val", "test"))
    df.loc[~df.model_eligible, "split_time"] = np.nan

    tr_inv = set(df.loc[df.split_random == "train", "investor"])
    te = df[df.split_random == "test"]
    cover = te.investor.isin(tr_inv).mean()
    log("- split_random (by company, 70/15/15): " + ", ".join(f"{k}={v}" for k, v in df.split_random.value_counts().items()))
    log(f"- Test rows whose investor also appears in train: {cover:.1%}")
    log("- split_time (<=2023 / 2024 / 2025-26): " + ", ".join(f"{k}={v}" for k, v in df.split_time.value_counts().items()))

    # excerpt leakage: text that names the lead investor must be masked before any NLP use
    def mask(row):
        t = row["news_excerpt"]
        if not isinstance(t, str):
            return np.nan
        for nm in {row["investor_raw"], row["investor"]}:
            if isinstance(nm, str) and nm:
                t = re.sub(re.escape(nm), "[INVESTOR]", t, flags=re.I)
        return t

    df["excerpt_masked"] = df.apply(mask, axis=1)
    leak = df["excerpt_masked"].fillna("").str.contains(r"\[INVESTOR\]").sum()
    log(f"- news_excerpt mentions the lead investor in {leak} rows -> use 'excerpt_masked', never the raw excerpt, as a model input")
    return df


# ------------------------------------------------------------------ 7. derived tables
def top_share(series, k=3):
    vc = series.value_counts()
    return "|".join(f"{i}:{v}" for i, v in vc.head(k).items())


def build_investor_profiles(df):
    d = df[df.investor_modelled].copy()
    g = d.groupby("investor")
    prof = pd.DataFrame({
        "n_deals": g.size(),
        "n_companies": g.company_id.nunique(),
        "first_year": g.year.min(),
        "last_year": g.year.max(),
        "deals_since_2023": g.year.apply(lambda s: (s >= 2023).sum()),
        "investor_type": g.investor_type.first(),
        "n_amounts_known": g.amount_known.sum(),
        "ticket_p25": g.amount_usd.quantile(.25),
        "ticket_median": g.amount_usd.median(),
        "ticket_p75": g.amount_usd.quantile(.75),
        "top_sectors": g.sector_group.apply(lambda s: top_share(s[s != "Unknown"])),
        "top_stages": g.stage_bucket.apply(lambda s: top_share(s[s != "Undisclosed"])),
        "top_cities": g.city_group.apply(lambda s: top_share(s[s != "Unknown"])),
    })
    sec = d[d.sector_known].groupby("investor").sector_group.apply(lambda s: s.value_counts(normalize=True).iloc[0])
    prof["top_sector_share"] = sec
    prof["is_specialist"] = prof.top_sector_share >= 0.5
    prof["is_active_recent"] = prof.deals_since_2023 >= 1
    prof = prof.round(3).sort_values("n_deals", ascending=False).reset_index()
    prof.to_csv(OUT / "investor_profiles.csv", index=False)

    for col, name in [("sector_group", "sector"), ("stage_bucket", "stage"), ("city_group", "city")]:
        mat = pd.crosstab(d.investor, d[col])
        mat.to_csv(OUT / f"investor_{name}_matrix.csv")
    return prof


def build_startup_pool(df):
    d = df.sort_values("announced_date")
    g = d.groupby("company_id")
    last = g.tail(1).set_index("company_id")
    pool = pd.DataFrame({
        "company_name": last.company_name, "company_domain": last.company_domain,
        "city": last.city_clean, "city_group": last.city_group,
        "sector_group": last.sector_group, "founded_year": last.founded_year,
        "latest_stage": last.stage_bucket, "latest_round_year": last.year,
        "latest_amount_usd": last.amount_usd,
        "n_rounds": g.size(),
        "total_raised_known_usd": g.amount_usd.sum(min_count=1),
        "investors_seen": g.investor.apply(lambda s: "|".join(dict.fromkeys(s.dropna()))),
        "news_mentions_total": g.news_mentions.sum(),
        "latest_excerpt_masked": last.excerpt_masked,
    }).reset_index()
    pool["age_now"] = 2026 - pool.founded_year
    pool.to_csv(OUT / "startups_pool.csv", index=False)
    return pool


# ------------------------------------------------------------------ 8. validation
def validate(df, prof, pool):
    log("\n## 7. Validation checks")
    checks = {
        "round_id unique": df.round_id.is_unique,
        "no negative company age": (df.company_age >= 0).all(),
        "all sectors mapped": df.sector_group.notna().all(),
        "all stages mapped": df.stage_bucket.notna().all(),
        "no company in more than one random split": df[df.split_random.notna()].groupby("company_id").split_random.nunique().max() == 1,
        "modelled investors all have >= MIN_DEALS": (prof.n_deals >= M.MIN_DEALS).all(),
        "pool has one row per company": pool.company_id.is_unique,
        "pool covers all companies": len(pool) == df.company_id.nunique(),
        "amounts positive": (df.amount_usd.dropna() > 0).all(),
    }
    for k, v in checks.items():
        log(f"- [{'PASS' if v else 'FAIL'}] {k}")
    assert all(checks.values()), "Validation failed - see report"


# ------------------------------------------------------------------ 9. figures
def make_figures(df, prof):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    df.groupby("year").size().plot.bar(ax=ax[0], color="#4C72B0"); ax[0].set_title("Rounds per year")
    df.sector_group.value_counts().plot.barh(ax=ax[1], color="#55A868"); ax[1].set_title("Rounds by sector group"); ax[1].invert_yaxis()
    df.stage_bucket.value_counts().reindex(M.STAGE_ORDER).plot.bar(ax=ax[2], color="#C44E52"); ax[2].set_title("Rounds by stage bucket")
    plt.tight_layout(); plt.savefig(REP / "figures" / "overview.png", dpi=130); plt.close()

    vc = df.investor.value_counts()
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    vc.head(20).plot.barh(ax=ax[0], color="#8172B2"); ax[0].invert_yaxis(); ax[0].set_title("Top 20 investors by deals")
    ax[1].hist(vc.values, bins=np.arange(1, 40), color="#CCB974"); ax[1].axvline(M.MIN_DEALS - 0.5, color="red", ls="--")
    ax[1].set_title(f"Deals per investor (red = min {M.MIN_DEALS})"); ax[1].set_xlabel("deals"); ax[1].set_ylabel("investors")
    plt.tight_layout(); plt.savefig(REP / "figures" / "investor_long_tail.png", dpi=130); plt.close()

    mat = pd.crosstab(df.loc[df.investor_modelled, "investor"], df.loc[df.investor_modelled, "sector_group"])
    top = mat.loc[prof.investor.head(25)]
    top = top.div(top.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(top.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(top.shape[1])); ax.set_xticklabels(top.columns, rotation=60, ha="right")
    ax.set_yticks(range(top.shape[0])); ax.set_yticklabels(top.index)
    ax.set_title("Investor x sector share (top 25 investors)"); plt.colorbar(im)
    plt.tight_layout(); plt.savefig(REP / "figures" / "investor_sector_heatmap.png", dpi=130); plt.close()


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (REP / "figures").mkdir(parents=True, exist_ok=True)
    df = load_and_audit()
    df = basic_clean(df)
    df = standardise_categoricals(df)
    df = numeric_features(df)
    df = clean_investors(df)
    df = modelling_flags(df)

    cols_front = ["round_id", "company_id", "company_name", "company_domain", "announced_date", "year",
                  "investor", "investor_type", "sector_group", "stage_bucket", "city_clean", "city_group",
                  "amount_usd", "amount_log10", "amount_bin", "founded_year", "company_age", "age_bin"]
    df = df[cols_front + [c for c in df.columns if c not in cols_front]]
    df.to_csv(OUT / "deals_clean.csv", index=False)
    df[df.model_eligible].to_csv(OUT / "deals_model.csv", index=False)

    prof = build_investor_profiles(df)
    pool = build_startup_pool(df)
    validate(df, prof, pool)
    make_figures(df, prof)

    log("\n## 8. Output files")
    for f in sorted(OUT.glob("*.csv")):
        log(f"- data/processed/{f.name}")
    (REP / "data_quality_report.md").write_text("# Data quality & preprocessing report\n\n" + "\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()
