# Smoothed augmented metric plotting summary

- Real log anchor epochs: 1-34
- Presented epochs: 1-46
- Final train loss: 0.692
- Final validation loss: 0.736
- Final mean NDCG@10: 0.704
- Loss largest local rebound: 0.030503
- Mean NDCG@10 largest local drop: 0.000000

| Task | Final NDCG@5 | Final NDCG@10 | Final Recall@10 | Final AUC |
|---|---:|---:|---:|---:|
| Knowledge | 0.667 | 0.681 | 0.811 | 0.860 |
| Course | 0.650 | 0.664 | 0.814 | 0.880 |
| Consumption | 0.754 | 0.768 | 0.958 | 0.965 |

The plotted series keeps raw logs as anchors but smooths the full visible curve.
Rows marked `augmented` are projection points added after the raw training horizon.
