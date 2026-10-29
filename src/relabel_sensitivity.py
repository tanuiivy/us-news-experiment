import sys
import numpy as np
import pandas as pd

T = float(sys.argv[1])
df = pd.read_csv("data/processed/us_features_basketA_sent.csv")
fut = df["future_pct_change"]
lab = pd.Series(np.nan, index=df.index, dtype=object)
lab[fut > T] = "upward"
lab[fut < -T] = "downward"
lab[(fut >= -T) & (fut <= T)] = "neutral"
before = df["trend_label"].value_counts(normalize=True).round(3).to_dict()
df["trend_label"] = lab
print("share at +-1.0:", before)
print(f"share at +-{T}:", df["trend_label"].value_counts(normalize=True).round(3).to_dict())
print("test split:", df[df["split"] == "test"]["trend_label"].value_counts(normalize=True).round(3).to_dict())
tag = str(T).replace(".", "")
df.to_csv(f"data/processed/us_features_basketA_sent_t{tag}.csv", index=False)
print("saved", f"data/processed/us_features_basketA_sent_t{tag}.csv")
