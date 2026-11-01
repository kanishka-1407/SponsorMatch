# Baseline results

Task: given a startup's sector, stage, city and amount, rank the 298 modelled investors; the true lead investor of the round is the single relevant item.

- Random split (by company): train=1546, val=318, test=387; test investors seen in train: 92.0%; tuned weights={'sector_group': 0.0, 'stage_bucket': 1.0, 'city_group': 0.0, 'amount_bin': 0.5}, prior_w=1.0, val nDCG@10=0.143
- Time split (train<=2023, test 2025-26): train=1831, val=218, test=202; test investors seen in train: 88.1%; tuned weights={'sector_group': 0.0, 'stage_bucket': 0.5, 'city_group': 0.0, 'amount_bin': 0.5}, prior_w=0.5, val nDCG@10=0.138

| split                                  | model                    |   Hit@1 |   Hit@5 |   Hit@10 |   Hit@20 |   MRR |   nDCG@10 |   Coverage@10 |   n_test |
|:---------------------------------------|:-------------------------|--------:|--------:|---------:|---------:|------:|----------:|--------------:|---------:|
| Random split (by company)              | Popularity               |   0.054 |   0.147 |    0.258 |    0.318 | 0.115 |     0.14  |            10 |      387 |
| Random split (by company)              | Affinity (equal weights) |   0.059 |   0.163 |    0.243 |    0.328 | 0.119 |     0.136 |           184 |      387 |
| Random split (by company)              | Affinity (tuned on val)  |   0.093 |   0.243 |    0.32  |    0.382 | 0.171 |     0.197 |            69 |      387 |
| Time split (train<=2023, test 2025-26) | Popularity               |   0.025 |   0.069 |    0.158 |    0.218 | 0.063 |     0.076 |            10 |      202 |
| Time split (train<=2023, test 2025-26) | Affinity (equal weights) |   0.054 |   0.139 |    0.198 |    0.243 | 0.108 |     0.119 |           163 |      202 |
| Time split (train<=2023, test 2025-26) | Affinity (tuned on val)  |   0.03  |   0.114 |    0.188 |    0.277 | 0.083 |     0.097 |            85 |      202 |