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

## 29 Oct: ablation design (fixed before running)
- A: technical only. B: technical + sentiment_score, sentiment_article_count, has_sentiment.
- Table: us_features_basketA_sent.csv. Same grid, selection rule, seed and split for both.
- Each model run once on test. No tuning after seeing test results.
- Comparison: paired date-bootstrap of B minus A, 95% interval.
- Known limitation: news thins out after 18 Dec 2023, so the last test days have has_sentiment=0.

## 29 Oct: pre-registered sensitivity checks (decided before running)
- S1: relabel with a ±1.5% neutral band, rerun A and B once each, same grid, rule, seed and split.
- S2: drop test days with no news coverage (after 18 Dec 2023) and re-evaluate the saved probabilities.
- Both are reported beside the primary ±1% result. Neither replaces it, and none is chosen for looking better.

## 29 Oct: sensitivity S3, full-text sentiment (decided before extraction or scoring)
- Reason: the NSE scored article text; this experiment scored headlines only. S3 tests whether that difference matters.
- Text: "Article_title. Article" for the same 71,778 rows, FinBERT on the first 512 tokens. Article text is cut at 4,000 characters at extraction (enough for 512 tokens).
- Rows with no usable article (under 200 characters) fall back to the headline score already computed. The share of fallbacks is reported.
- Feasibility rule: if fewer than 50% of rows have usable article text, S3 is reported as not feasible and not run.
- Everything else unchanged: timing rule, features, grid, selection rule, seed, split, +-1.0% band. Model A is unchanged and is reused.
- Run B once on the new sentiment columns. Compare B(text) with A and with B(headline) using the paired bootstrap.
- Report the result whatever it shows. No second attempt or alternative scoring after seeing it.

## 29 Oct: validity check V1 (decided before running)
- Question: do FinBERT scores relate to the same-day return? This checks the scores carry information. It is not a prediction test and is not used to select anything.
- Method: mean score per ticker and article date (no lag) versus that day's close-to-close change (pct_change_1d). Spearman correlation and mean return by score tercile.
- Report whatever it shows.
- V1 result (29 Oct): 16,026 ticker-days. Spearman 0.155 between mean score and same-day return. Mean same-day return by score tercile: -0.485% (lowest), +0.190% (middle), +0.517% (highest). Scores carry information about same-day moves. Association is not prediction, and some headlines describe the price move itself.

## 29 Oct: social media (X) not tested, recorded as future work
- X API: no free read access; historical archive only on the highest paid tiers; the default plan reaches back 7 days. Not feasible for the 2023 test period.
- Public dataset checked: Kaggle stock tweets, 2021-09-30 to 2022-09-30, about 81,000 tweets, about 20% duplicate text, Tesla 46% and TSM 14% of rows. No overlap with the 2023 test period, so it could only support a separate pilot.
- Licence, timestamp precision and ticker list were not verified.
- Decision: not pursued in this project. Reported as a limitation and as future work.

## 29 Oct: event study (Task 19), decided before running
- Events: ticker-days with at least 3 articles, split by the daily mean sentiment score into the top 10% ("good news") and bottom 10% ("bad news"). News dated on non-trading days is not used.
- Abnormal return: the stock's daily return minus the basket's average return that day.
- Windows: trading days -1, 0 (event day), +1, +2, +3. Only days +1 to +3 bear on predictability. Day 0 is the same-day reaction.
- Second view: days where a ticker's article count is in its own top 10% (and at least 3): extra size of move (mean absolute abnormal return) on days 0, +1, +2 versus all other days.
- Intervals: 1,000 resamples of whole dates, seed 42, 95% percentile intervals.
- Descriptive only: no model, no tuning, no choosing among variants. All of it is reported, including anything that shows nothing.
- About 1 in 20 intervals will exclude zero by chance. A single isolated hit is not read as a finding.
- Event study result (29 Oct): 900 good-news and 900 bad-news events. Event-day abnormal return +0.38% (good) and -0.96% (bad); good minus bad +1.34 pp (1.12 to 1.55). Days +1 to +3: good minus bad +0.14, -0.17, +0.04, all intervals include zero. Day -1: good minus bad +0.43 (0.18 to 0.65), probably headlines reporting the prior day's move (interpretation, not tested). Article surges: extra absolute move +1.16 pp (event day), +0.59 (day +1), +0.20 (day +2), all intervals above zero; not a causal claim.
- S3 extraction (30 Oct): 71,778 of 71,778 rows, aligned with the headlines file. Usable article text 86.1% overall (2019 58%, 2020 75%, 2021 to 2023 100%). Feasibility rule met (50% or more). Rows without usable text use the headline score. The test period (2023) is fully covered; the fallback applies only to training years.
