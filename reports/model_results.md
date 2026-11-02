# Neural network model results (Version 3)

## Model

A startup-to-investor ranking model using categorical feature embeddings combined with normalized numeric features, followed by a two-layer feed-forward neural network.

Categorical features: sector_group, stage_bucket, city_group, amount_bin

Numeric features: amount_log10, company_age

Missing amount_log10 values are imputed using the training median and accompanied by a missing-value indicator.

Model selection and early stopping use validation nDCG@10.

## Results

| split                                  | model             |   Hit@1 |   Hit@5 |   Hit@10 |   Hit@20 |    MRR |   nDCG@10 |   Coverage@10 |   n_test |   best_val_nDCG@10 |
|:---------------------------------------|:------------------|--------:|--------:|---------:|---------:|-------:|----------:|--------------:|---------:|-------------------:|
| Random split (by company)              | Neural Network v3 |  0.0853 |  0.2248 |   0.3127 |   0.3773 | 0.1636 |    0.1892 |            98 |      387 |             0.134  |
| Time split (train<=2023, test 2025-26) | Neural Network v3 |  0.0495 |  0.1386 |   0.1782 |   0.2574 | 0.1006 |    0.1086 |            72 |      202 |             0.1484 |