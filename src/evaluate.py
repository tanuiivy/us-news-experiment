import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score

CLASS_NAMES = ["downward", "neutral", "upward"]   # codes 0, 1, 2, same as NSE

def all_metrics(y, p):
    pred = p.argmax(axis=1)
    mv = y != 1                       # direction is only defined on rows that truly moved
    denom = p[mv, 0] + p[mv, 2]
    # 0.5 where the denominator is 0: no directional information for that row
    score_up = np.divide(p[mv, 2], denom, out=np.full(denom.shape, 0.5), where=denom > 0)
    acc = accuracy_score(y, pred)
    return {
        "accuracy": acc,
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y, pred),
        "auc_macro": roc_auc_score(y, p, multi_class="ovr", average="macro"),
        "auc_direction": roc_auc_score((y[mv] == 2).astype(int), score_up),
        "auc_neutral_vs_moving": roc_auc_score((y == 1).astype(int), p[:, 1]),
        "acc_minus_floor": acc - (y == 1).mean(),   # floor = always predicting neutral
    }

def floors(y):
    pred = np.full(len(y), 1)
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
            "balanced_acc": 1 / 3, "auc_macro": 0.5, "auc_direction": 0.5,
            "auc_neutral_vs_moving": 0.5, "acc_minus_floor": 0.0}

def date_bootstrap(dates, y, probs, seed=42, n_boot=1000):
    """Resample whole trading dates (as NSE Week 4) and return estimates with 95% CIs."""
    dates = np.asarray(dates)
    udates = np.unique(dates)
    rows_by_date = [np.flatnonzero(dates == d) for d in udates]
    rng = np.random.default_rng(seed)
    boot = {m: [] for m in probs}
    for _ in range(n_boot):
        pick = rng.integers(0, len(udates), len(udates))
        idx = np.concatenate([rows_by_date[i] for i in pick])
        for m, p in probs.items():
            boot[m].append(all_metrics(y[idx], p[idx]))
    boot = {m: pd.DataFrame(v) for m, v in boot.items()}
    fl = floors(y)
    records = []
    for m, p in probs.items():
        for k, v in all_metrics(y, p).items():
            lo, hi = np.percentile(boot[m][k], [2.5, 97.5])
            records.append({"model": m, "metric": k, "estimate": v,
                            "ci_low": lo, "ci_high": hi, "floor": fl[k]})
    return pd.DataFrame(records), boot
