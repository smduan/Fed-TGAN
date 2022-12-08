# Fed-TGAN
The unofficial implementation of paper [Fed-TGAN: Federated learning framework for synthesizing tabular data](https://arxiv.org/pdf/2108.07927.pdf)

# Usage Example

Run this repo:
```
python main.py
```

## Evaluation results on Adult dataset
The average Jensen-Shannon Divergence (JSD) and the average Wasserstein Distance (WD)
| |AVG_JSD|AVG_WD|
|--|--|--|
|train dataset| 0.0087|0.0018|
|CTGAN|0.1076|0.0657|
|Fed-TGAN (beta=100)|0.039|0.049|







