import numpy as np
import pandas as pd

feat = pd.read_csv("data/processed/us_features_basketA.csv")
feat["Date"] = pd.to_datetime(feat["Date"])
cal = feat["Date"].drop_duplicates().sort_values().to_numpy()

s = pd.read_csv("data/processed/news_scores_basketA.csv")
tu = pd.to_datetime(s["Date"], utc=True)
tn = tu.dt.tz_convert("America/New_York")
date_only = tu.dt.strftime("%H:%M:%S") == "00:00:00"
print(f"date-only rows: {date_only.mean():.4f}")

utc_day = tu.dt.normalize().dt.tz_localize(None)
ny_day = tn.dt.normalize().dt.tz_localize(None)
cand_dateonly = utc_day + pd.Timedelta(days=1)                         # strictly after the article date
cand_real = ny_day + pd.to_timedelta((tn.dt.hour >= 16).astype(int), unit="D")  # 16:00 ET cutoff
cand = cand_dateonly.where(date_only, cand_real)

idx = np.searchsorted(cal, cand.to_numpy(), side="left")
keep = idx < len(cal)
print(f"dropped (no later trading day): {(~keep).sum()}")
s = s[keep].copy()
s["eff"] = cal[idx[keep]]

agg = (s.groupby(["Stock_symbol", "eff"])
         .agg(sentiment_score=("sentiment_score", "mean"),
              sentiment_article_count=("sentiment_score", "size"))
         .reset_index().rename(columns={"Stock_symbol": "Code", "eff": "Date"}))
assert agg["sentiment_article_count"].sum() == len(s)

out = feat.merge(agg, on=["Code", "Date"], how="left")
out["sentiment_score"] = out["sentiment_score"].fillna(0.0)
out["sentiment_article_count"] = out["sentiment_article_count"].fillna(0).astype(int)
out["has_sentiment"] = (out["sentiment_article_count"] > 0).astype(int)

print("rows:", len(out), "| ticker-days with news:", round(out["has_sentiment"].mean(), 3))
print(out.groupby("split")["has_sentiment"].mean().round(3).to_string())
print("min coverage by ticker:", out.groupby("Code")["has_sentiment"].mean().round(2).sort_values().head(3).to_dict())

out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
out.to_csv("data/processed/us_features_basketA_sent.csv", index=False)
print("saved data/processed/us_features_basketA_sent.csv")
