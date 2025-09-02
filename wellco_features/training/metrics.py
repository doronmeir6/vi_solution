import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def precision_at_frac(y_true, scores, frac: float) -> float:
    n = len(scores); k = max(1, int(np.floor(frac * n)))
    idx = np.argsort(-scores)[:k]
    return float(np.mean(y_true[idx]))

def lift_at_frac(y_true, scores, frac: float) -> float:
    base = np.mean(y_true)
    pa = precision_at_frac(y_true, scores, frac)
    return float(pa / base) if base > 0 else float("nan")

def metric_dict(y_true, scores):
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr": float(average_precision_score(y_true, scores)),
        "p@5%": precision_at_frac(y_true, scores, 0.05),
        "p@10%": precision_at_frac(y_true, scores, 0.10),
        "lift@5%": lift_at_frac(y_true, scores, 0.05),
        "lift@10%": lift_at_frac(y_true, scores, 0.10),
        "base_rate": float(np.mean(y_true)),
    }
