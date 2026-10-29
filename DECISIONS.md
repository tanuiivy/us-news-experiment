# Decision log: US news experiment

In order, 29 Oct 2026:
1. Rule amended before any model results: common stocks only (no ETFs), NASDAQ-listed throughout 2019-2023. WMT (NYSE during the window) and QQQ (ETF) excluded.
2. Basket A fixed by the rule "steadiest news coverage" (FNSPID nasdaq file). Technical-only baseline run once on it: outputs/us_baseline_basketA.csv.
3. The request to include the largest NASDAQ names (Apple, Nvidia, Microsoft, Amazon) was made before basket A results existed. Plan: apply rule v2 to All_external (NASDAQ-listed, common stock, at least 200 articles in every year, top 15 by market capitalisation), one file only, no mixing.
4. Basket A results did not influence step 3. Both baskets will be reported; no tuning after any test result.
