# Startup–Investor Matcher: Data Preprocessing

Turns `data/raw/india_startup_funding_sample.csv` (5,006 funding rounds, 2015–2026) into clean,
model-ready tables for the matching model and the two-page app.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/preprocess.py        # writes data/processed/*, reports/*
pytest -q                       # 7 checks
```

## Pipeline (src/preprocess.py)
1. Load + audit (nulls, duplicates, dates)
2. Basic cleaning (whitespace, unicode, domains, company_id from domain)
3. Sector (48 → 14 groups), stage (24 → 7 buckets), city (typos → canonical → 11 groups)
4. Amount (unknown handling, log10, bins) and company age
5. Investor cleaning (junk removal, name normalisation, manual aliases, fuzzy review list)
6. Eligibility flags, company-level train/val/test splits, excerpt leakage masking
7. Derived tables: investor profiles + matrices, startup pool
8. Automated validation + figures + report

## Outputs (data/processed)
| File | Use |
|---|---|
| deals_clean.csv | All 5,006 rounds, cleaned, nothing dropped, with flags |
| deals_model.csv | Rows eligible for training (known investor with ≥3 deals, known sector) |
| investor_profiles.csv | One row per modelled investor: deals, tickets, top sectors/stages/cities |
| investor_{sector,stage,city}_matrix.csv | Investor × category deal counts (rule-based baseline) |
| startups_pool.csv | One row per company (for the investor-side page) |
| investor_alias_map.csv | Audit trail of merged investor names |

## Rules for the modelling team
- Train/evaluate with `deals_model.csv` using `split_random` (by company) or `split_time`.
- Never use `news_excerpt` directly as an input; use `excerpt_masked` (investor names hidden).
- Never use `investors_seen` from `startups_pool.csv` as a feature (it contains the target).
- Use `amount_log10`, not raw `amount_usd` (heavy-tailed).
- Source: dataset on Hugging Face via fliarbi.com (see `fliar_profile_url`). Check the licence and cite it in the report.

## Pitch connection requests

Install API dependencies with `pip install -r requirements.txt`, then run `uvicorn api:app --reload`. The matching API keeps `/rank` and `/match/startups`; connection requests use the SQLite database `sponsormatch.sqlite3` (override with `SPONSORMATCH_DB`). Register accounts with `POST /pitch/register` using role `startup` (requires `company`) or `investor` (requires `name`), then authenticate with `POST /pitch/login` and send the returned bearer token in `Authorization: Bearer <token>`.

Pitch routes: `POST /pitch/generate` creates an editable profile-based draft; `POST /pitch/requests` sends it with an optional deck URL and score/reasons; `GET /pitch/inbox` lists investor requests; `GET /pitch/requests` and `GET /pitch/requests/{id}` view requests; `POST /pitch/requests/{id}/respond` accepts `accept`, `decline`, or `request_info`; `POST /pitch/requests/{id}/withdraw` withdraws a pending outgoing request. `GET /pitch/notifications` lists notifications and `POST /pitch/notifications/{id}/read` marks one read. Request messages are limited to 300 characters, outgoing requests to five per rolling seven days, and pending duplicate pairs are rejected. Peer contact email appears only after acceptance.
