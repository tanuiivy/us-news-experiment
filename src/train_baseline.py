import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from evaluate import date_bootstrap

SEED = 42
TOL = 0.005   # rule fixed in advance: simplest setting within 0.005 of the best mean macro AUC
FEATURES = ["close_to_ma", "macd_line_rel", "day_range_pct", "pct_change_1d", "RSI_14", "rel_volume"]
import sys
SENT = len(sys.argv) > 3 and sys.argv[3] == "sent"
if SENT:
    FEATURES = FEATURES + ["sentiment_score", "sentiment_article_count", "has_sentiment"]
LABELS = {"downward": 0, "neutral": 1, "upward": 2}   # same coding as NSE

in_path, out_path = sys.argv[1], sys.argv[2]
df = pd.read_csv(in_path, parse_dates=["Date"])
df = df[df["trend_label"].notna()].copy()    # unlabelled last rows are dropped before modelling
df["y"] = df["trend_label"].map(LABELS)

train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
# The last training date's label uses the first validation date's price, so it is not used to fit
fit_df = train_df[train_df["Date"] < train_df["Date"].max()]

# Time-ordered CV folds over unique training dates, with a one-date gap
dates = pd.DatetimeIndex(np.sort(train_df["Date"].unique()))
pos = train_df["Date"].map({d: i for i, d in enumerate(dates)}).to_numpy()
folds = []
for tr_d, va_d in TimeSeriesSplit(n_splits=4, gap=1).split(dates):
    folds.append((np.flatnonzero(pos <= tr_d[-1]),
                  np.flatnonzero((pos >= va_d[0]) & (pos <= va_d[-1]))))

def make_xgb(depth, n_est):
    return XGBClassifier(objective="multi:softprob", max_depth=depth, n_estimators=n_est,
                         learning_rate=0.1, min_child_weight=20,
                         random_state=SEED, n_jobs=-1, verbosity=0)

def make_rf(depth, leaf, n_est):
    return RandomForestClassifier(max_depth=depth, min_samples_leaf=leaf, n_estimators=n_est,
                                  class_weight="balanced", random_state=SEED, n_jobs=-1)

def cv_auc(make_model, weighted):
    scores = []
    for tr, va in folds:
        Xtr, ytr = train_df.loc[tr, FEATURES], train_df.loc[tr, "y"]
        Xva, yva = train_df.loc[va, FEATURES], train_df.loc[va, "y"]
        m = make_model()
        if weighted:   # XGBoost gets balanced sample weights from this fold's training rows
            m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
        else:
            m.fit(Xtr, ytr)
        scores.append(roc_auc_score(yva, m.predict_proba(Xva), multi_class="ovr", average="macro"))
    return float(np.mean(scores))

xgb_res = pd.DataFrame([{"max_depth": d, "n_estimators": n,
                         "auc": cv_auc(lambda d=d, n=n: make_xgb(d, n), True)}
                        for d in [1, 2, 3, 4, 6] for n in [100, 300]])
elig = xgb_res[xgb_res["auc"] >= xgb_res["auc"].max() - TOL]
xgb_pick = elig.sort_values(["max_depth", "n_estimators"]).iloc[0]

rf_res = pd.DataFrame([{"max_depth": d, "min_samples_leaf": l, "n_estimators": n,
                        "auc": cv_auc(lambda d=d, l=l, n=n: make_rf(d, l, n), False)}
                       for d in [3, 6, 10, None] for l in [20, 100] for n in [200, 500]])
rf_res["depth_order"] = rf_res["max_depth"].apply(lambda d: 999 if pd.isna(d) else d)  # None = deepest
elig = rf_res[rf_res["auc"] >= rf_res["auc"].max() - TOL]
rf_pick = elig.sort_values(["depth_order", "n_estimators", "min_samples_leaf"],
                           ascending=[True, True, False]).iloc[0]

X_fit, y_fit = fit_df[FEATURES], fit_df["y"]
xgb = make_xgb(int(xgb_pick["max_depth"]), int(xgb_pick["n_estimators"]))
xgb.fit(X_fit, y_fit, sample_weight=compute_sample_weight("balanced", y_fit))
depth = None if pd.isna(rf_pick["max_depth"]) else int(rf_pick["max_depth"])
rf = make_rf(depth, int(rf_pick["min_samples_leaf"]), int(rf_pick["n_estimators"]))
rf.fit(X_fit, y_fit)

y_arr = test_df["y"].to_numpy()
probs = {"xgboost": xgb.predict_proba(test_df[FEATURES]),
         "random_forest": rf.predict_proba(test_df[FEATURES])}
results, _ = date_bootstrap(test_df["Date"].to_numpy(), y_arr, probs, seed=SEED)
results.to_csv(out_path, index=False)

print("XGBoost picked:", dict(xgb_pick.drop("auc")), "| CV AUC", round(xgb_pick["auc"], 3))
print("Forest picked:", dict(rf_pick.drop(["auc", "depth_order"])), "| CV AUC", round(rf_pick["auc"], 3))
print("test rows:", len(test_df), "| test dates:", test_df["Date"].nunique(),
      "| neutral share (the floor):", round((y_arr == 1).mean(), 3))
keep = ["accuracy", "balanced_acc", "auc_macro", "auc_direction", "auc_neutral_vs_moving", "acc_minus_floor"]
print(results[results["metric"].isin(keep)].round(3).to_string(index=False))

import os
tag = ("sent" if SENT else "tech") + os.environ.get("RUN_SUFFIX", "")
cols = {"Code": test_df["Code"], "Date": test_df["Date"], "y": test_df["y"]}
for m in probs:
    for k in range(probs[m].shape[1]):
        cols[f"{m}_p{k}"] = probs[m][:, k]
pd.DataFrame(cols).to_csv(f"outputs/test_probs_{tag}.csv", index=False)
print("saved outputs/test_probs_" + tag + ".csv")
