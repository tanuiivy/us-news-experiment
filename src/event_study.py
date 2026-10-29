import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OFFS = [-1, 0, 1, 2, 3]       # trading days relative to the event day
MIN_ART = 3                   # minimum articles on the day for a sentiment event
Q = 0.10                      # top and bottom decile of the daily mean score
NBOOT, SEED = 1000, 42

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
})

# ---------- Data ----------
f = pd.read_csv("data/processed/us_features_basketA_sent.csv")[["Code", "Date", "pct_change_1d"]]
f["Date"] = pd.to_datetime(f["Date"])
f = f.sort_values(["Code", "Date"]).reset_index(drop=True)
# Abnormal return: the stock's daily return minus the basket's average return that day
f["ar"] = f["pct_change_1d"] - f.groupby("Date")["pct_change_1d"].transform("mean")
for o in OFFS:
    f[f"ar_{o}"] = f.groupby("Code")["ar"].shift(-o)

n = pd.read_csv("data/processed/news_scores_basketA.csv")
n["Date"] = n["Date"].str[:10]
g = n.groupby(["Stock_symbol", "Date"])["sentiment_score"].agg(score="mean", count="size").reset_index()
g.columns = ["Code", "Date", "score", "count"]
g["Date"] = pd.to_datetime(g["Date"])
f = f.merge(g, on=["Code", "Date"], how="left")
f["count"] = f["count"].fillna(0)
print(f"ticker-days: {len(f)} | with news on that trading day: {(f['count'] > 0).mean():.1%}")

def boot_by_date(dates, make_stat, nboot=NBOOT):
    """Resample whole dates with replacement and return the stat's 2.5 and 97.5 percentiles."""
    ud = np.unique(dates)
    rows = [np.flatnonzero(dates == d) for d in ud]
    rng = np.random.default_rng(SEED)
    out = []
    for _ in range(nboot):
        pick = rng.integers(0, len(ud), len(ud))
        out.append(make_stat(np.concatenate([rows[i] for i in pick])))
    out = np.array(out)
    return np.nanpercentile(out, 2.5, axis=0), np.nanpercentile(out, 97.5, axis=0)

# ---------- E1: good-news and bad-news days ----------
ev = f[f["count"] >= MIN_ART].reset_index(drop=True)
hi, lo = ev["score"].quantile(1 - Q), ev["score"].quantile(Q)
good = (ev["score"] >= hi).to_numpy()
bad = (ev["score"] <= lo).to_numpy()
arr = ev[[f"ar_{o}" for o in OFFS]].to_numpy()
dates = ev["Date"].to_numpy()
print(f"events: {good.sum()} good-news, {bad.sum()} bad-news (score cut-offs {lo:.2f} and {hi:.2f})")

def stat_e1(idx):
    gi, bi = idx[good[idx]], idx[bad[idx]]
    mg, mb = np.nanmean(arr[gi], axis=0), np.nanmean(arr[bi], axis=0)
    return np.concatenate([mg, mb, mg - mb])

point = np.concatenate([np.nanmean(arr[good], axis=0), np.nanmean(arr[bad], axis=0),
                        np.nanmean(arr[good], axis=0) - np.nanmean(arr[bad], axis=0)])
cl, ch = boot_by_date(dates, stat_e1)
k = len(OFFS)
rows = []
for gi, name in enumerate(["good news", "bad news", "good minus bad"]):
    for j, o in enumerate(OFFS):
        i = gi * k + j
        rows.append({"group": name, "offset": o, "mean_abnormal_return_pct": round(point[i], 4),
                     "ci_low": round(cl[i], 4), "ci_high": round(ch[i], 4),
                     "n_events": int(good.sum() if gi == 0 else bad.sum() if gi == 1 else good.sum() + bad.sum())})
e1 = pd.DataFrame(rows)
e1.to_csv("outputs/event_study_sentiment.csv", index=False)
print(e1.to_string(index=False))

# ---------- E2: news volume and the size of the move ----------
thr = f.groupby("Code")["count"].transform(lambda s: s.quantile(0.90))
shock = ((f["count"] >= MIN_ART) & (f["count"] >= thr)).to_numpy()
OFF2 = [0, 1, 2]
absar = np.abs(f[[f"ar_{o}" for o in OFF2]].to_numpy())
dates2 = f["Date"].to_numpy()

def stat_e2(idx):
    s, o_ = idx[shock[idx]], idx[~shock[idx]]
    return np.nanmean(absar[s], axis=0) - np.nanmean(absar[o_], axis=0)

p2 = np.nanmean(absar[shock], axis=0) - np.nanmean(absar[~shock], axis=0)
c2l, c2h = boot_by_date(dates2, stat_e2)
e2 = pd.DataFrame({"offset": OFF2,
                   "shock_mean_abs_ar": np.nanmean(absar[shock], axis=0).round(4),
                   "other_mean_abs_ar": np.nanmean(absar[~shock], axis=0).round(4),
                   "difference": p2.round(4), "ci_low": c2l.round(4), "ci_high": c2h.round(4),
                   "n_shock_days": int(shock.sum())})
e2.to_csv("outputs/event_study_attention.csv", index=False)
print(e2.to_string(index=False))

# ---------- Figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ax = axes[0]
for name, col, dx in [("good news", BLUE, -0.07), ("bad news", ORANGE, 0.07)]:
    d = e1[e1["group"] == name]
    x = np.array(OFFS) + dx
    ax.vlines(x, d["ci_low"], d["ci_high"], color=col, linewidth=2)
    ax.plot(x, d["mean_abnormal_return_pct"], "o", color=col, markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.2, label=f"{name} (n = {int(d['n_events'].iloc[0])})")
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(OFFS)
ax.set_xticklabels(["day -1", "event day", "day +1", "day +2", "day +3"], fontsize=9)
ax.set_ylabel("Average abnormal return (%)", fontsize=9)
ax.set_title("Price reaction around good-news and bad-news days", loc="left", fontsize=10.5, color=INK, pad=10)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

ax = axes[1]
x = np.arange(len(OFF2))
ax.vlines(x, e2["ci_low"], e2["ci_high"], color=BLUE, linewidth=2)
ax.plot(x, e2["difference"], "o", color=BLUE, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)
ax.axhline(0, color=INK2, linewidth=1)
ax.set_xticks(x)
ax.set_xticklabels(["event day", "day +1", "day +2"], fontsize=9)
ax.set_ylabel("Extra size of move (percentage points)", fontsize=9)
ax.set_title(f"Days with a surge in articles (n = {int(shock.sum())}): bigger moves?", loc="left", fontsize=10.5, color=INK, pad=10)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
fig.text(0.01, 0.01, "Abnormal return = stock return minus the basket's average that day. Lines are 95% intervals from resampling whole dates.\n"
         "An interval that includes zero means no detectable difference.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig("outputs/fig_event_study.png", dpi=200)
print("saved outputs/event_study_sentiment.csv, event_study_attention.csv, fig_event_study.png")
