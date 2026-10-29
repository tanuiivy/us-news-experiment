import sys
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

IN, PROBS = sys.argv[1], sys.argv[2]
S = sys.argv[3] if len(sys.argv) > 3 else ""
SEED = 42
TECH = ["close_to_ma", "macd_line_rel", "day_range_pct", "pct_change_1d", "RSI_14", "rel_volume"]
SENT = ["sentiment_score", "sentiment_article_count", "has_sentiment"]
FEATURES = TECH + SENT
LABELS = {"downward": 0, "neutral": 1, "upward": 2}

df = pd.read_csv(IN, parse_dates=["Date"])
df = df[df["trend_label"].notna()].copy()
df["y"] = df["trend_label"].map(LABELS)
train_df = df[df["split"] == "train"].reset_index(drop=True)
test_df = df[df["split"] == "test"].reset_index(drop=True)
fit_df = train_df[train_df["Date"] < train_df["Date"].max()]

# Settings picked by the baseline's selection rule (printed there as depth 1, 100 trees)
xgb = XGBClassifier(objective="multi:softprob", max_depth=1, n_estimators=100,
                    learning_rate=0.1, min_child_weight=20,
                    random_state=SEED, n_jobs=-1, verbosity=0)
xgb.fit(fit_df[FEATURES], fit_df["y"], sample_weight=compute_sample_weight("balanced", fit_df["y"]))

ref = pd.read_csv(PROBS)
assert (ref["Code"].values == test_df["Code"].values).all(), "test rows differ from saved probabilities"
p = xgb.predict_proba(test_df[FEATURES])
diff = np.abs(p - ref[["xgboost_p0", "xgboost_p1", "xgboost_p2"]].to_numpy()).max()
print(f"refit check: max abs diff vs saved probabilities = {diff:.2e}")
assert diff < 1e-3, "refit does not match the evaluated model, stop"

sv = shap.TreeExplainer(xgb).shap_values(test_df[FEATURES])
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (rows, features, classes)
imp = np.abs(sv).mean(axis=0)
tab = pd.DataFrame(imp, index=FEATURES, columns=["downward", "neutral", "upward"])
tab["overall"] = tab.mean(axis=1)
tab["is_sentiment"] = [f in SENT for f in FEATURES]
tab = tab.sort_values("overall", ascending=False)
tab["rank"] = range(1, len(tab) + 1)
print(tab.round(4).to_string())
print("share of total importance from sentiment features:",
      round(tab.loc[tab["is_sentiment"], "overall"].sum() / tab["overall"].sum(), 4))
tab.to_csv(f"outputs/us_shap_importance{S}.csv")

order = tab.sort_values("overall")
colors = ["#c0504d" if s else "#4f81bd" for s in order["is_sentiment"]]
plt.figure(figsize=(7, 4))
plt.barh(order.index, order["overall"], color=colors)
plt.xlabel("Mean absolute SHAP value (average over the three classes)")
plt.title("NASDAQ basket A: feature importance, technical + sentiment model")
plt.tight_layout()
plt.savefig(f"outputs/us_shap_importance{S}.png", dpi=150)
print("saved outputs/us_shap_importance" + S + ".csv and .png")
