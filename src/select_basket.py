import sys
import pandas as pd
import yfinance as yf

counts_path = sys.argv[1]
N_TARGET = 15
YEARS = [2019, 2020, 2021, 2022, 2023]
EXPECTED_DAYS = 1258                    # trading days 2019-01-02 to 2023-12-29, as seen for AAPL
NASDAQ_CODES = {"NMS", "NGM", "NCM"}    # Yahoo's codes for the three NASDAQ tiers

c = pd.read_csv(counts_path)
c["year"] = c["year"].astype(int)
wide = c.pivot_table(index="Stock_symbol", columns="year", values="n_articles",
                     aggfunc="sum", fill_value=0)
wide = wide.reindex(columns=YEARS, fill_value=0)   # a missing year counts as 0 articles
wide["min_year"] = wide[YEARS].min(axis=1)         # the steadiness measure
wide["total"] = wide[YEARS].sum(axis=1)
ranked = wide.sort_values(["min_year", "total"], ascending=False)

accepted, rows = [], []
for t, r in ranked.iterrows():
    if len(accepted) == N_TARGET:
        break
    try:
        tk = yf.Ticker(t)
        ex = tk.info.get("exchange")
        days = len(tk.history(start="2019-01-01", end="2024-01-01", auto_adjust=False))
    except Exception:
        ex, days = None, 0
    ok = (ex in NASDAQ_CODES) and (days == EXPECTED_DAYS)
    print(f"{t:8} min_year={int(r['min_year']):>6} exchange={ex} days={days} -> {'ACCEPT' if ok else 'skip'}")
    rows.append({"ticker": t, "min_year": int(r["min_year"]), "total": int(r["total"]),
                 "exchange": ex, "days": days, "accepted": ok})
    if ok:
        accepted.append(t)

pd.DataFrame(rows).to_csv("outputs/basket_selection.csv", index=False)
print("\nBASKET:", " ".join(accepted))
