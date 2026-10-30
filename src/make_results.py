import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
})

RUNS = [
    ("Primary (+-1.0% band)", "outputs/us_paired_B_minus_A.csv", 0.518),
    ("Late December removed (+-1.0%)", "outputs/us_paired_B_minus_A_until.csv", None),
    ("Wider band (+-1.5%)", "outputs/us_paired_B_minus_A_t15.csv", 0.677),
    ("Full-text sentiment (+-1.0%)", "outputs/us_paired_B_minus_A_text.csv", 0.518),
]
MODEL_NAME = {"xgboost": "XGBoost", "random_forest": "Random Forest"}
MODEL_COLOR = {"xgboost": BLUE, "random_forest": ORANGE}
METRIC_NAME = {"accuracy": "Accuracy", "balanced_acc": "Balanced accuracy", "auc_macro": "AUC (macro)",
               "auc_direction": "AUC (direction)", "auc_neutral_vs_moving": "AUC (neutral vs moving)"}

# ---------- Table ----------
frames = []
for run, path, floor in RUNS:
    d = pd.read_csv(path)
    d.insert(0, "run", run)
    d["always_neutral_accuracy"] = floor
    frames.append(d)
tab = pd.concat(frames, ignore_index=True).rename(columns={
    "A_tech": "technical_only", "B_sent": "with_sentiment", "B_minus_A": "difference"})
tab["model"] = tab["model"].map(MODEL_NAME)
tab.to_csv("outputs/results_table.csv", index=False)

order = ["auc_neutral_vs_moving", "auc_direction", "auc_macro", "balanced_acc", "accuracy"]
raw = pd.concat(frames, ignore_index=True)
lines = ["| Run | Model | Metric | Technical only | With sentiment | Difference (95% interval) |",
         "|---|---|---|---|---|---|"]
for run, _, _ in RUNS:
    for m in ["xgboost", "random_forest"]:
        for k in order:
            r = raw[(raw["run"] == run) & (raw["model"] == m) & (raw["metric"] == k)].iloc[0]
            flag = " *" if r["excludes_zero"] else ""
            lines.append(f"| {run} | {MODEL_NAME[m]} | {METRIC_NAME[k]} | {r['A_tech']:.3f} | {r['B_sent']:.3f} | "
                         f"{r['B_minus_A']:+.3f} ({r['ci_low']:+.3f}, {r['ci_high']:+.3f}){flag} |")
lines.append("")
lines.append("* interval excludes zero. Always-neutral accuracy: 0.518 (+-1.0%), 0.677 (+-1.5%).")
open("outputs/results_table.md", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# ---------- Figure 1: SHAP, both markets ----------
NICE = {"day_range_pct": "Day range", "close_to_ma": "Close vs 10-day average", "pct_change_1d": "1-day change",
        "rel_volume": "Relative volume", "macd_line_rel": "MACD", "RSI_14": "RSI (14)",
        "sentiment_score": "Sentiment score", "sentiment_article_count": "Article count", "has_sentiment": "Has news"}

def prep(path):
    t = pd.read_csv(path, index_col=0)
    t["pct"] = 100 * t["overall"] / t["overall"].sum()
    return t.sort_values("pct")

panels = [("NASDAQ basket A", "87% of ticker-days have news", prep("outputs/us_shap_importance.csv")),
          ("NSE", "0.2% of ticker-days have news", prep("outputs/nse_shap_importance.csv"))]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for ax, (name, sub, t) in zip(axes, panels):
    cols = [ORANGE if s else BLUE for s in t["is_sentiment"]]
    y = np.arange(len(t))
    ax.barh(y, t["pct"], color=cols, height=0.62, edgecolor=SURFACE, linewidth=1.5)
    for yi, v in zip(y, t["pct"]):
        ax.text(v + 0.9, yi, f"{v:.1f}%", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE[i] for i in t.index], fontsize=9)
    ax.set_xlim(0, 72)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Share of total importance (%)", fontsize=9)
    ax.set_title(f"{name}  ({sub})", loc="left", fontsize=10.5, color=INK, pad=10)
axes[0].legend(handles=[Patch(color=BLUE, label="Technical features"), Patch(color=ORANGE, label="Sentiment features")],
               loc="lower right", frameon=False, fontsize=9)
fig.text(0.01, 0.01, "Importance is the mean absolute SHAP value averaged over the three classes, shown as a share of each model's total.",
         fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("outputs/fig_shap_both_markets.png", dpi=200)
plt.close(fig)

# ---------- Figure 2: paired differences ----------
metrics = ["auc_neutral_vs_moving", "auc_direction"]
fig, axes = plt.subplots(2, len(RUNS), figsize=(14, 5.2))
for r, k in enumerate(metrics):
    sub = raw[raw["metric"] == k]
    lim = float(np.abs(sub[["ci_low", "ci_high"]].to_numpy()).max()) * 1.35
    for c, (run, _, _) in enumerate(RUNS):
        ax = axes[r, c]
        for yi, m in [(1, "xgboost"), (0, "random_forest")]:
            row = sub[(sub["run"] == run) & (sub["model"] == m)].iloc[0]
            col = MODEL_COLOR[m]
            ax.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=col, linewidth=2, solid_capstyle="round")
            ax.plot(row["B_minus_A"], yi, marker="o", markersize=8, color=col, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3)
            v = row["B_minus_A"]
            txt = "0.0000" if abs(v) < 0.00005 else f"{v:+.4f}"
            ax.text(v, yi + 0.22, txt, ha="center", va="bottom", fontsize=8, color=INK2, zorder=4,
                    bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, 1.7)
        ax.set_yticks([1, 0])
        ax.set_yticklabels(["XGBoost", "Random Forest"] if c == 0 else ["", ""], fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if r == 0:
            ax.set_title(run, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlabel(f"Change in {METRIC_NAME[k]}", fontsize=8.5)
fig.text(0.01, 0.01, "Points show with-sentiment minus technical-only on the test set; lines are 95% date-bootstrap intervals. "
         "An interval that includes zero means no detectable difference.", fontsize=8.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig("outputs/fig_paired_differences.png", dpi=200)
plt.close(fig)
print("saved outputs/results_table.csv, results_table.md, fig_shap_both_markets.png, fig_paired_differences.png")
