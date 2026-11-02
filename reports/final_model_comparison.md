# Final Model Comparison

## Startup-to-Investor Ranking

This report compares the baseline recommendation models with the neural network model.

## Models Compared

- **Popularity**: Recommends investors based on their frequency in the training data.

- **Affinity (equal weights)**: Ranks investors based on similarity between startup and investor-related features using equal feature weights.

- **Affinity (tuned on validation)**: Uses feature weights tuned using the validation set.

- **Neural Network v3**: Uses categorical feature embeddings and normalized numeric features followed by a feed-forward neural network.

## Evaluation Metrics

- **Hit@1**: Correct investor appears at rank 1.
- **Hit@5**: Correct investor appears in the top 5.
- **Hit@10**: Correct investor appears in the top 10.
- **Hit@20**: Correct investor appears in the top 20.
- **MRR**: Mean Reciprocal Rank; rewards higher-ranked correct recommendations.
- **nDCG@10**: Measures ranking quality within the top 10.
- **Coverage@10**: Number of unique investors appearing in the top-10 recommendations.

## Final Results

| split                                  | model                    |   Hit@1 |   Hit@5 |   Hit@10 |   Hit@20 |    MRR |   nDCG@10 |   Coverage@10 |   n_test |
|:---------------------------------------|:-------------------------|--------:|--------:|---------:|---------:|-------:|----------:|--------------:|---------:|
| Random split (by company)              | Affinity (tuned on val)  |  0.093  |  0.2429 |   0.3204 |   0.3824 | 0.1714 |    0.1974 |            69 |      387 |
| Random split (by company)              | Neural Network v3        |  0.0853 |  0.2248 |   0.3127 |   0.3773 | 0.1636 |    0.1892 |            98 |      387 |
| Random split (by company)              | Popularity               |  0.0543 |  0.1473 |   0.2584 |   0.3178 | 0.1154 |    0.1397 |            10 |      387 |
| Random split (by company)              | Affinity (equal weights) |  0.0594 |  0.1628 |   0.2429 |   0.3282 | 0.119  |    0.1364 |           184 |      387 |
| Time split (train<=2023, test 2025-26) | Affinity (equal weights) |  0.0545 |  0.1386 |   0.198  |   0.2426 | 0.1082 |    0.1191 |           163 |      202 |
| Time split (train<=2023, test 2025-26) | Neural Network v3        |  0.0495 |  0.1386 |   0.1782 |   0.2574 | 0.1006 |    0.1086 |            72 |      202 |
| Time split (train<=2023, test 2025-26) | Affinity (tuned on val)  |  0.0297 |  0.1139 |   0.1881 |   0.2772 | 0.0834 |    0.097  |            85 |      202 |
| Time split (train<=2023, test 2025-26) | Popularity               |  0.0248 |  0.0693 |   0.1584 |   0.2178 | 0.0627 |    0.0755 |            10 |      202 |

## Conclusion

The validation-tuned affinity model leads on the random company split: nDCG@10 is 0.1974, compared with 0.1892 for the neural network. On the time split, the equal-weight affinity model leads nDCG@10 at 0.1191, compared with 0.1086 for the neural network. The neural model has greater random-split top-10 investor coverage (98 vs. 69 for tuned affinity) but does not lead on ranking quality in either split.

These are single-run results from different evaluation regimes; they do not establish that one method is universally better. The app therefore blends neural and affinity rank positions rather than presenting neural softmax probabilities as match likelihoods. `src/multi_seed_evaluation.py` adds five-seed random-split reporting for the neural and hybrid systems. Its output should be regenerated before making claims about seed-level stability. The final API checkpoint is trained on train plus validation data after selection; the test split stays excluded.