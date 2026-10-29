# Decision log: US news experiment

In order, 29 Oct 2026:
1. Rule amended before any model results: common stocks only (no ETFs), NASDAQ-listed throughout 2019-2023. WMT (NYSE during the window) and QQQ (ETF) excluded.
2. Basket A fixed by the rule "steadiest news coverage" (FNSPID nasdaq file). Technical-only baseline run once on it: outputs/us_baseline_basketA.csv.
3. The request to include the largest NASDAQ names (Apple, Nvidia, Microsoft, Amazon) was made before basket A results existed. Plan: apply rule v2 to All_external (NASDAQ-listed, common stock, at least 200 articles in every year, top 15 by market capitalisation), one file only, no mixing.
4. Basket A results did not influence step 3. Both baskets will be reported; no tuning after any test result.
5. Rule v2 parameters fixed before seeing All_external counts: at least 200 articles in every year 2019-2023; busiest year at most 10x the thinnest year; NASDAQ-listed common stock (quoteType EQUITY); WMT excluded (NYSE during window); full 2019-2023 price history; top 15 by current market cap. If fewer than 15 qualify, basket A stands and the gap is reported as a limitation.
6. Outcome of rule v2: All_external has no articles after 11 Jun 2020, so no ticker had coverage in every year 2019-2023 (0 candidates). Fallback applied: basket A is the primary basket. The mega-cap coverage gap (AAPL, MSFT, NVDA, AMZN) is reported as a limitation.

## 29 Oct: URLs for basket A
- Headlines file excludes Url by design. A second filtered pass (src/fnspid_basket_urls.py) recovered Url for the same 71,778 rows, same order.
- Verified: row counts equal, titles match on every row, 0 missing URLs.
- URL file is git-ignored (news_raw in name). Scoring uses headlines only, so FinBERT does not depend on it. Scores join to URLs by row_id.
- FinBERT run on GPU with half precision (autocast), batch 128. Scores may differ from full precision in the 3rd or 4th decimal place.

## 29 Oct: news timing rule (amended before any sentiment result)
- Found: 99.7% of FNSPID rows are date-only (00:00:00 UTC), so no time of day is known.
- First rule (16:00 ET cutoff for all rows) would have put date-only news on its own date, which risks look-ahead (e.g. after-close earnings).
- Final rule: date-only article counts for the first trading day strictly after its date; the ~0.3% with a real time use the 16:00 ET cutoff.
- Limitation to report: sentiment reaches the model one trading day after publication.
- No model with sentiment had been run when this was decided.
