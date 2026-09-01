"""Generate notebooks/02_vol_forecast_eval.ipynb — the hourly H=1 dev-set
volatility-forecast evaluation (PLAN §6.1).

    .venv/bin/python notebooks/_build_vol_eval.py

Reads the cached panel (store/panel_hourly_H1_gk.parquet) — no Kronos compute.
"""
from __future__ import annotations

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md("""# Kronos vs classical baselines — hourly volatility, 1-session horizon

**Dev set:** 30 US large-caps, 378 sessions (2024-07 → 2025-12), 11,340 forecasts,
fully out of sample (Kronos pretraining ends ~2024-06). Target: next session's
intraday realized vol (Garman–Klass). All baselines walk-forward.

The probe (8 names, pooled Spearman, no proper HAR) *overstated* Kronos. With
real baselines and hypothesis tests the picture is soberer — and more useful.
""")
    code("""import os
if "src" not in os.listdir("."):
    os.chdir("..")""")
    code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src.evaluate import assemble, vol_scorecard, dm_vs, incremental_r2, mcs, calibration, by_cut
from src.metrics import qlike
p = assemble("hourly", 1)
p = p[~p.in_lockbox].copy()
MODELS = ["har","kronos_recal","ewma94","rolling22","kronos","rw","gjr","garch"]
len(p), p.ticker.nunique(), p.origin.min().date(), p.origin.max().date()""")

    md("## 1 · Scorecard — HAR-RV wins, Kronos ≈ EWMA")
    code("""sc = vol_scorecard(p, MODELS); sc.round(4)""")
    code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4))
o = sc.sort_values("QLIKE")
c = ["crimson" if "kronos" in m else "#888" for m in o.index]
ax[0].barh(range(len(o)), o["QLIKE"], color=c); ax[0].set_yticks(range(len(o))); ax[0].set_yticklabels(o.index)
ax[0].invert_yaxis(); ax[0].set_title("QLIKE (lower = better)"); ax[0].axvline(o["QLIKE"].min(), ls=":", c="k", lw=.8)
ax[1].scatter(sc["MZ_slope"], sc["QLIKE"], c=c, s=60)
for m in sc.index: ax[1].annotate(m, (sc.loc[m,"MZ_slope"], sc.loc[m,"QLIKE"]), fontsize=8)
ax[1].axvline(1.0, ls=":", c="k", lw=.8); ax[1].set_xlabel("Mincer–Zarnowitz slope (1 = unbiased)"); ax[1].set_ylabel("QLIKE")
ax[1].set_title("calibration vs accuracy"); plt.tight_layout(); plt.show()""")

    md("## 2 · Diebold–Mariano & the Model Confidence Set")
    code("""print(dm_vs(p, ref="ewma94", models=[m for m in MODELS if m!="ewma94"]).round(4).to_string())
print()
print("Model Confidence Set (QLIKE, 90%):", mcs(p, MODELS))""")
    md("""HAR-RV is the **sole** member of the 90% MCS — every other model, Kronos
included, is statistically excluded as inferior. Raw Kronos vs EWMA is a
Diebold–Mariano tie.""")

    md("## 3 · The one place Kronos wins — incremental information")
    code("""rows = []
for signal in ["kronos","kronos_recal"]:
    for over in ["ewma94","har"]:
        r = incremental_r2(p, add=signal, base=over)
        rows.append({"signal":signal,"over":over,"R2_base":r[f"r2_{over}"],
                     "gain":r["gain"],"ci_lo":r["gain_ci95"][0],"ci_hi":r["gain_ci95"][1]})
inc = pd.DataFrame(rows); inc.round(4)""")
    code("""sub = inc[inc["signal"] == "kronos"].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7,4))
x = np.arange(len(sub))
ax.bar(x, sub["R2_base"], label="baseline alone", color="#bbb")
ax.bar(x, sub["gain"], bottom=sub["R2_base"], label="+ Kronos", color="crimson")
ax.errorbar(x, sub["R2_base"]+sub["gain"],
            yerr=[sub["gain"]-sub["ci_lo"], sub["ci_hi"]-sub["gain"]],
            fmt="none", ecolor="k", capsize=4)
ax.set_xticks(x); ax.set_xticklabels(list(sub["over"])); ax.set_ylabel("log-RV R²")
ax.set_title("Kronos adds orthogonal vol info — even on top of HAR"); ax.legend(); plt.show()
print(inc.round(4).to_string(index=False))""")
    md("""Kronos lifts log-RV R² by **+0.065 over EWMA** and **+0.013 over HAR**
(the best model) — small, but both bootstrap CIs cleanly exclude zero. It is not
a better forecaster; it sees ~1 R² point of volatility that price-history models
miss.""")

    md("## 4 · Calibration — overconfident, recalibration half-fixes it")
    code("""cal = calibration(p); cal""")
    code("""from src.metrics import pit_exact
dp = p.dropna(subset=["realised_ret"]); dp = dp[dp.ret_paths.notna()]
pit = pit_exact(dp["realised_ret"].to_numpy(), list(dp["ret_paths"]))
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].hist(pit, bins=20, color="crimson", alpha=.8, edgecolor="white")
ax[0].axhline(len(pit)/20, ls="--", c="k", lw=1, label="calibrated")
ax[0].set_title(f"return PIT — KS p={cal['return_PIT']['ks_p']:.0e}  (U-shaped ⇒ intervals too tight)")
ax[0].set_xlabel("PIT"); ax[0].legend()
covs = {"90% interval": (cal["return_cov_90"], .90), "50% interval": (cal["return_cov_50"], .50),
        "90% RV interval": (cal.get("rv_cov_90", np.nan), .90)}
ax[1].bar(range(len(covs)), [v[0] for v in covs.values()], color="crimson", alpha=.8)
for i,(k,v) in enumerate(covs.items()): ax[1].axhline(v[1], xmin=i/len(covs)+.05, xmax=(i+1)/len(covs)-.05, c="k")
ax[1].set_xticks(range(len(covs))); ax[1].set_xticklabels(covs.keys(), fontsize=9)
ax[1].set_title("empirical coverage vs nominal (black bars)"); ax[1].set_ylim(0,1); plt.tight_layout(); plt.show()""")

    md("## 5 · By regime — Kronos is *not* better when it's turbulent")
    code("""reg = by_cut(p, "vol_q", MODELS); reg.round(4)""")
    code("""r = reg[["kronos_recal","har","ewma94","garch"]].drop(columns=[]).iloc[:, :]
r = reg.loc[["calm","low","elevated","turbulent"], ["kronos_recal","ewma94","har","garch"]]
r.plot.bar(figsize=(11,4)); plt.ylabel("QLIKE"); plt.title("QLIKE by realised-vol quartile"); plt.xticks(rotation=0); plt.show()""")
    md("""The probe's "edge concentrates in turbulent regimes" was an artefact of
pooled cross-name Spearman. On QLIKE, Kronos is **worst** in the turbulent
quartile; GARCH (mean-reverting) wins there.""")

    md("## 6 · By sector — the actual signal: Kronos helps where vol is *flow-driven*")
    code("""sec = by_cut(p, "sector", MODELS)
sec["kronos_edge_vs_ewma"] = sec["ewma94"] - sec["kronos_recal"]   # positive = Kronos better
sec = sec.sort_values("kronos_edge_vs_ewma", ascending=False)
sec[["n","kronos_recal","ewma94","har","kronos_edge_vs_ewma"]].round(4)""")
    code("""fig, ax = plt.subplots(figsize=(11,4))
c = ["crimson" if v>0 else "#888" for v in sec["kronos_edge_vs_ewma"]]
ax.bar(sec.index, sec["kronos_edge_vs_ewma"], color=c)
ax.axhline(0, c="k", lw=.8); ax.set_ylabel("EWMA QLIKE − Kronos_recal QLIKE")
ax.set_title("Kronos's edge over EWMA by sector  (positive = Kronos better)")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()""")
    md("""**Kronos beats EWMA in Tech, Comm, Consumer Discretionary — and loses in
Health, Utilities, Materials.**

The pattern is *what drives short-horizon vol in each sector*:

- **Tech / growth / consumer disc:** vol is **flow-driven** — momentum, gap
  continuation, options-gamma, retail. It shows up in candle shapes and volume,
  which is exactly what Kronos reads.
- **Health, utilities, materials:** vol is **event-driven** (FDA rulings,
  litigation, guidance cuts, drug-pricing politics, rate moves) or just sleepy.
  Discrete catalysts are invisible to a price-only model — hence the big UNH /
  LLY misses in the Health bucket.

Kronos does best exactly where catalysts matter least and price structure
matters most. That is the same limitation seen in the probe (the TSLA-earnings /
DeepSeek misses), viewed cross-sectionally.""")

    md("## 7 · One name, over time")
    code("""tk = "NVDA"
g = p[p.ticker==tk].sort_values("origin")
fig, ax = plt.subplots(figsize=(13,4))
ax.plot(g.origin, g.realised, c="k", lw=1.5, label="realised RV")
ax.plot(g.origin, g.kronos_recal, c="crimson", lw=1, alpha=.85, label="Kronos (recal)")
ax.plot(g.origin, g.har, c="tab:blue", lw=1, alpha=.85, label="HAR-RV")
ax.set_title(f"{tk} — next-session realised vol vs forecasts"); ax.legend(); plt.show()""")

    md("""---
## What this establishes

1. **Kronos-small is not a good standalone equity vol forecaster.** HAR-RV beats
   it on QLIKE, RMSE, calibration, and is the sole Model Confidence Set member.
   Raw Kronos ≈ EWMA; recalibration lifts it just past EWMA, still short of HAR.
2. **It carries a small, robust slice of orthogonal information** — +0.013 log-RV
   R² even on top of HAR (CI [0.010, 0.015]).
3. **Its intervals are overconfident** — 68% coverage for a nominal 90% — and an
   expanding-window affine recalibration only half-closes the gap.
4. **The edge is where vol is flow-driven, not news-driven** — Tech/Comm/ConsDisc,
   not Health/Utilities/Materials. Consistent with a price-only model being blind
   to discrete catalysts.

## Still open
- GARCH/GJR baselines are fit on close-to-close returns (target is intraday) —
  unfair, needs an intraday-returns refit. Won't change the HAR verdict.
- Lockbox (2026-01 → 2026-08) untouched.
- L=250 / S=50 sensitivity checks.
- Daily-horizon study.
""")
    return nb


def main() -> None:
    nb = build()
    out = __import__("pathlib").Path(__file__).parent / "02_vol_forecast_eval.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
