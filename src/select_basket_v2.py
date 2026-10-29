import sys
import time
import pandas as pd
import yfinance as yf

counts_path, out_path = sys.argv[1], sys.argv[2]
N_TARGET = 15
YEARS = [2019, 2020, 2021, 2022, 2023]
FLOOR = 200            # articles in every year
MAX_RATIO = 10         # busiest year at most 10x the thinnest, to limit news-density drift
EXPECTED_DAYS = 1258   # full 2019-2023 price history
NASDAQ_CODES = {"NMS", "NGM", "NCM"}
EXCLUDE = {"WMT": "NYSE-listed during 2019-2023"}

c = pd.read_csv(counts_path)
c["year"] = c["year"].astype(int)
w = c.pivot_table(index="Stock_symbol", columns="year", values="n_articles",
                  aggfunc="sum", fill_value=0).reindex(columns=YEARS, fill_value=0)
w["min_year"] = w[YEARS].min(axis=1)
w["max_year"] = w[YEARS].max(axis=1)
cand = w[(w["min_year"] >= FLOOR) & (w["max_year"] <= MAX_RATIO * w["min_year"])]
print(f"{len(cand)} tickers pass the coverage floor and stability guard", flush=True)

rows = []
for t in cand.index:
    try:
        info = yf.Ticker(t).info
        rows.append({"ticker": t, "exchange": info.get("exchange"), "type": info.get("quoteType"),
                     "market_cap": info.get("marketCap") or 0,
                     "min_year": int(cand.loc[t, "min_year"]), "max_year": int(cand.loc[t, "max_year"])})
    except Exception:
        pass
    time.sleep(0.2)    # be gentle with Yahoo
info_df = pd.DataFrame(rows)
pool = info_df[info_df["exchange"].isin(NASDAQ_CODES) & (info_df["type"] == "EQUITY")]
pool = pool[~pool["ticker"].isin(EXCLUDE)].sort_values("market_cap", ascending=False).reset_index(drop=True)
print(f"{len(pool)} are NASDAQ-listed common stocks; checking price history from the largest down", flush=True)

accepted = []
for _, r in pool.iterrows():
    if len(accepted) == N_TARGET:
        break
    days = len(yf.Ticker(r["ticker"]).history(start="2019-01-01", end="2024-01-01", auto_adjust=False))
    ok = days == EXPECTED_DAYS
    print(f"{r['ticker']:8} cap=${r['market_cap']/1e9:8.1f}B min_year={r['min_year']:>6} "
          f"max_year={r['max_year']:>6} days={days} -> {'ACCEPT' if ok else 'skip'}", flush=True)
    if ok:
        accepted.append(r["ticker"])

pool["accepted"] = pool["ticker"].isin(accepted)
pool.to_csv(out_path, index=False)
print("\nBASKET:", " ".join(accepted), "| size", len(accepted))
