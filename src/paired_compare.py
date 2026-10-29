import sys, os
S = os.environ.get("RUN_SUFFIX", "")
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from evaluate import all_metrics

until = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. 2023-12-18 for sensitivity S2
A = pd.read_csv(f"outputs/test_probs_tech{S}.csv")
B = pd.read_csv(f"outputs/test_probs_sent{S}.csv")
assert (A[["Code", "Date", "y"]].values == B[["Code", "Date", "y"]].values).all(), "rows differ"
if until:
    keep = (A["Date"] <= until).to_numpy()
    A, B = A[keep].reset_index(drop=True), B[keep].reset_index(drop=True)

y = A["y"].to_numpy()
dates = A["Date"].to_numpy()
udates = np.unique(dates)
rows_by_date = [np.flatnonzero(dates == d) for d in udates]
print(f"rows: {len(A)} | dates: {len(udates)}" + (f" | up to {until}" if until else ""))

def P(df, m):
    return df[[f"{m}_p0", f"{m}_p1", f"{m}_p2"]].to_numpy()

METRICS = ["accuracy", "balanced_acc", "auc_macro", "auc_direction", "auc_neutral_vs_moving"]
rng = np.random.default_rng(42)
rows = []
for m in ["xgboost", "random_forest"]:
    pa, pb = P(A, m), P(B, m)
    a0, b0 = all_metrics(y, pa), all_metrics(y, pb)
    rng = np.random.default_rng(42)
    diffs = {k: [] for k in METRICS}
    for _ in range(1000):
        pick = rng.integers(0, len(udates), len(udates))
        idx = np.concatenate([rows_by_date[i] for i in pick])
        ra, rb = all_metrics(y[idx], pa[idx]), all_metrics(y[idx], pb[idx])
        for k in METRICS:
            diffs[k].append(rb[k] - ra[k])
    for k in METRICS:
        lo, hi = np.percentile(diffs[k], [2.5, 97.5])
        rows.append({"model": m, "metric": k, "A_tech": round(a0[k], 4), "B_sent": round(b0[k], 4),
                     "B_minus_A": round(b0[k] - a0[k], 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "excludes_zero": bool(lo > 0 or hi < 0)})
out = pd.DataFrame(rows)
print(out.to_string(index=False))
out.to_csv("outputs/us_paired_B_minus_A" + ("_until" if until else "") + S + ".csv", index=False)
