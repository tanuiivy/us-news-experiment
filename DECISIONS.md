# Decision log: US news experiment

In order, 29 Oct 2026:
1. Rule amended before any model results: common stocks only (no ETFs), NASDAQ-listed throughout 2019-2023. WMT (NYSE during the window) and QQQ (ETF) excluded.
2. Basket A fixed by the rule "steadiest news coverage" (FNSPID nasdaq file). Technical-only baseline run once on it: outputs/us_baseline_basketA.csv.
3. The request to include the largest NASDAQ names (Apple, Nvidia, Microsoft, Amazon) was made before basket A results existed. Plan: apply rule v2 to All_external (NASDAQ-listed, common stock, at least 200 articles in every year, top 15 by market capitalisation), one file only, no mixing.
4. Basket A results did not influence step 3. Both baskets will be reported; no tuning after any test result.
5. Rule v2 parameters fixed before seeing All_external counts: at least 200 articles in every year 2019-2023; busiest year at most 10x the thinnest year; NASDAQ-listed common stock (quoteType EQUITY); WMT excluded (NYSE during window); full 2019-2023 price history; top 15 by current market cap. If fewer than 15 qualify, basket A stands and the gap is reported as a limitation.
6. Outcome of rule v2: All_external has no articles after 11 Jun 2020, so no ticker had coverage in every year 2019-2023 (0 candidates). Fallback applied: basket A is the primary basket. The mega-cap coverage gap (AAPL, MSFT, NVDA, AMZN) is reported as a limitation.
