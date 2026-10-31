import os, sys
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predict_explain import (make_features, _load_data, _load_model, DEFAULT_DATA_PATH,
                             DEFAULT_MODEL_PATH, DEFAULT_OUTPUT_DIR, FEATURES_SENTIMENT, CLASS_NAMES)

SENT = ["sentiment_score", "sentiment_article_count", "has_sentiment"]
df = _load_data(DEFAULT_DATA_PATH)
full = pd.concat([df[["Code", "Date", "trend_label"]], make_features(df)], axis=1)

# Same chronological rule as the NSE holdout: global 70/15/15 by unique dates, test = last 15%
dates = np.sort(full["Date"].unique())
n = len(dates)
test_start = dates[int(n * 0.85)]
test = full[(full["Date"] >= test_start) & full["trend_label"].notna()]
print(f"test rows: {len(test)} | test dates: {test['Date'].nunique()} | tickers: {test['Code'].nunique()}")
print(f"test range: {test['Date'].min().date()} to {test['Date'].max().date()}")

model = _load_model(DEFAULT_MODEL_PATH)
X = test[FEATURES_SENTIMENT].astype(float)
sv = shap.TreeExplainer(model).shap_values(X)
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else np.asarray(sv)   # (rows, features, classes)
tab = pd.DataFrame(np.abs(sv).mean(axis=0), index=FEATURES_SENTIMENT, columns=CLASS_NAMES)
tab["overall"] = tab.mean(axis=1)
tab["is_sentiment"] = [f in SENT for f in FEATURES_SENTIMENT]
tab = tab.sort_values("overall", ascending=False)
tab["rank"] = range(1, len(tab) + 1)
print(tab.round(4).to_string())
print("share of total importance from sentiment features:",
      round(tab.loc[tab["is_sentiment"], "overall"].sum() / tab["overall"].sum(), 4))

os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
tab.to_csv(os.path.join(DEFAULT_OUTPUT_DIR, "nse_shap_importance.csv"))
order = tab.sort_values("overall")
plt.figure(figsize=(7, 4))
plt.barh(order.index, order["overall"], color=["#c0504d" if s else "#4f81bd" for s in order["is_sentiment"]])
plt.xlabel("Mean absolute SHAP value (average over the three classes)")
plt.title("NSE: feature importance, technical + sentiment model")
plt.tight_layout()
plt.savefig(os.path.join(DEFAULT_OUTPUT_DIR, "nse_shap_importance.png"), dpi=150)
print("saved nse_shap_importance.csv and .png in ml/outputs")
