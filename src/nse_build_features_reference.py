import numpy as np
import pandas as pd

def build_features(df):
    g = df.groupby("Code")["Day Price"]
    ma10 = g.transform(lambda x: x.rolling(10, min_periods=10).mean())
    ema_f = g.transform(lambda x: x.ewm(span=12, adjust=False, min_periods=12).mean())
    ema_s = g.transform(lambda x: x.ewm(span=26, adjust=False, min_periods=26).mean())
    macd = ema_f - ema_s
    pct1 = g.transform(lambda x: x.pct_change() * 100)

    out = pd.DataFrame(index=df.index)
    out["close_to_ma"] = df["Day Price"] / ma10 - 1
    out["macd_line_rel"] = macd / df["Day Price"]
    out["day_range_pct"] = (df["Day High"] - df["Day Low"]) / df["Day Price"]
    out["pct_change_1d"] = pct1

    delta = g.diff()
    gain, loss = delta.clip(lower=0), -delta.clip(upper=0)
    avg_gain = gain.groupby(df["Code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    avg_loss = loss.groupby(df["Code"]).transform(lambda s: s.rolling(14, min_periods=14).mean())
    rsi = 100 - 100 / (1 + avg_gain / avg_loss.replace(0, np.nan))
    out["RSI_14"] = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)

    vol0 = df["Volume"].fillna(0)
    past_mean = vol0.groupby(df["Code"]).transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean())
    out["rel_volume"] = (vol0 / past_mean).where(past_mean > 0)

    fut = g.transform(lambda x: (x.shift(-1) - x) / x * 100)
    label = pd.Series(np.nan, index=df.index, dtype=object)
    label[fut > 1] = "upward"
    label[fut < -1] = "downward"
    label[(fut >= -1) & (fut <= 1)] = "neutral"
    out["trend_label"] = label
    out["future_pct_change"] = fut
    return out
