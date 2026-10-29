import sys
import numpy as np
import pandas as pd
import yfinance as yf
from nse_build_features_reference import build_features  # NSE function, copied unchanged

START, END = "2019-01-01", "2024-01-01"   # END is exclusive, so the last row is 29 Dec 2023
THRESHOLD = 1.0   # percent, same as NSE. Change only for the pre-decided sensitivity run

def load_prices(tickers):
    frames = []
    for t in tickers:
        # auto_adjust=False keeps Close split-adjusted but not dividend-adjusted, like NSE's raw close
        h = yf.Ticker(t).history(start=START, end=END, auto_adjust=False)
        h = h.rename(columns={"Close": "Day Price", "High": "Day High", "Low": "Day Low"})
        h["Code"] = t
        h["Date"] = h.index.tz_localize(None).normalize()
        frames.append(h[["Code", "Date", "Day Price", "Day High", "Day Low", "Volume"]])
    df = pd.concat(frames, ignore_index=True)
    # build_features does not sort, so the order must be Code then Date
    return df.sort_values(["Code", "Date"]).reset_index(drop=True)

def main(tickers, out_path):
    df = load_prices(tickers)
    feats = build_features(df)

    # Rebuild the label from future_pct_change so the threshold is a parameter
    fut = feats["future_pct_change"]
    label = pd.Series(np.nan, index=df.index, dtype=object)
    label[fut > THRESHOLD] = "upward"
    label[fut < -THRESHOLD] = "downward"
    label[(fut >= -THRESHOLD) & (fut <= THRESHOLD)] = "neutral"
    feats["trend_label"] = label

    out = pd.concat([df[["Code", "Date"]], feats], axis=1)

    # Global chronological 70/15/15 split by unique date, same arithmetic as NSE
    dates = sorted(out["Date"].unique())
    n = len(dates)
    train_cut = dates[int(n * 0.70) - 1]
    val_cut = dates[int(n * 0.85) - 1]
    out["split"] = np.where(out["Date"] <= train_cut, "train",
                   np.where(out["Date"] <= val_cut, "validation", "test"))
    out.to_csv(out_path, index=False)

    print("rows:", len(out), "| tickers:", out["Code"].nunique())
    print("dates per split:", out.groupby("split")["Date"].nunique().to_dict())
    print("label shares:", out["trend_label"].value_counts(normalize=True).round(3).to_dict())
    print("NaN labels:", int(out["trend_label"].isna().sum()), "(expect 1 per ticker)")

if __name__ == "__main__":
    main(sys.argv[2:], sys.argv[1])
