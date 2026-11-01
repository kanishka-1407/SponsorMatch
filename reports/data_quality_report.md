# Data quality & preprocessing report

## 1. Raw audit
- Rows: 5006, columns: 19
- Duplicate round_id: 0; exact duplicate rows: 0
- Missing values: city=25, sector=39, amount_usd=975, lead_investor=1004, news_source=2118, news_url=2118, news_excerpt=74, fliar_profile_url=13
- Date range: 2015-01-04 to 2026-06-11

## 2. Basic cleaning
- Unparseable dates: 0; year column disagrees with date: 0
- Unique companies (by domain): 2163; by name: 2162

## 3. Sector, stage, city
- Sector: 48 raw labels -> 14 groups; unknown rows: 39
- Stage: 24 raw labels -> 7 buckets
- City: 127 raw spellings -> 99 cities -> 11 groups

## 4. Amount and company age
- Amount known: 4031 / 5006; suspiciously low (<$5K): 5; unverified: 71
- Outliers are kept (a $1B round is real); models should use log10 amount.
- Negative ages: 0; founded_year == 2000 (likely clipped at lower bound): 48 rows

## 5. Investors
- Blank lead_investor: 1004 rows
- Junk / non-identifiable entries removed: 70 rows (67 distinct)
- Distinct investor spellings: 1790 -> canonical investors: 1728
- Alias map saved: 44 canonical names absorbed >1 spelling
- Near-duplicate investor pairs flagged for manual review: 9 (reports/investor_near_duplicates_for_review.csv)

## 6. Modelling eligibility and splits
- Investors with >= 3 deals: 298; eligible rows: 2251 (1098 companies)
- split_random (by company, 70/15/15): train=1546, test=387, val=318
- Test rows whose investor also appears in train: 92.0%
- split_time (<=2023 / 2024 / 2025-26): train=1831, val=218, test=202
- news_excerpt mentions the lead investor in 2737 rows -> use 'excerpt_masked', never the raw excerpt, as a model input

## 7. Validation checks
- [PASS] round_id unique
- [PASS] no negative company age
- [PASS] all sectors mapped
- [PASS] all stages mapped
- [PASS] no company in more than one random split
- [PASS] modelled investors all have >= MIN_DEALS
- [PASS] pool has one row per company
- [PASS] pool covers all companies
- [PASS] amounts positive

## 8. Output files
- data/processed/deals_clean.csv
- data/processed/deals_model.csv
- data/processed/investor_alias_map.csv
- data/processed/investor_city_matrix.csv
- data/processed/investor_profiles.csv
- data/processed/investor_sector_matrix.csv
- data/processed/investor_stage_matrix.csv
- data/processed/startups_pool.csv