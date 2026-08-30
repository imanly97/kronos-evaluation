"""Generate notebooks/kronos_baseline.ipynb — how good is the raw Kronos forecast,
before any agent, over the whole replay window.

    .venv/bin/python notebooks/_build_kronos_baseline.py

Assumes the forecast cache is built and graded:
    python -m src.forecasts
    python -m src.grade
"""
from __future__ import annotations

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md("""# Kronos baseline — the forecaster on its own

Before the triage agent, the critic, or the memory loop: **how good is the raw
Kronos-small forecast** on this task (next-day close-to-close, 40 US large-caps,
graded on the `asof` bar)?

### Contamination check

Kronos-small's pretraining data ends **~June 2024** (Shi et al., *Kronos: A
Foundation Model for the Language of Financial Markets*, arXiv:2508.02739 — test
period starts July 2024). The replay window is **2025-05-01 → 2025-08-01**, about
**11 months out of sample**. Kronos has never seen these price paths; the
forecasts are genuine predictions, not lookups.
""")

    code("""import os
if "src" not in os.listdir("."):        # so `import src` works from notebooks/
    os.chdir("..")""")
    code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt
from src import baseline
j = baseline._frame()
rep = baseline.report()
print(rep["window"], "|", rep["forecast_days"], "forecast-days,",
      rep["sessions"], "sessions,", rep["tickers"], "tickers")""")

    md("""## 1 · Calibration (PIT)

If the forecast distribution were calibrated, the actual close's quantile
position (`actual_quantile`) would be **uniform on [0, 1]**. Excess mass in the
end bins = the intervals are too tight (overconfident).""")
    code("""cal = baseline.calibration(j); cal""")
    code("""q = j["actual_quantile"].to_numpy()
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(q, bins=np.linspace(0,1,11), edgecolor="white")
ax[0].axhline(len(q)/10, ls="--", c="k", lw=1, label="calibrated (uniform)")
ax[0].set_title("PIT histogram"); ax[0].set_xlabel("actual_quantile"); ax[0].legend()
xs = np.sort(q)
ax[1].plot(xs, np.arange(1,len(xs)+1)/len(xs), label="empirical CDF")
ax[1].plot([0,1],[0,1], ls="--", c="k", lw=1, label="uniform")
ax[1].set_title(f"PIT CDF vs uniform (KS = {cal['ks_vs_uniform']})"); ax[1].legend()
plt.tight_layout(); plt.show()""")
    code("""print(f"90% envelope (q05-q95) covers: {cal['frac_in_05_95']:.0%}   (target 90%)")
print(f"50% band     (q25-q75) covers: {cal['frac_in_25_75']:.0%}   (target 50%)")
print("=> the distribution is OVERCONFIDENT — real moves land in the tails "
      "more often than the model implies.")""")

    md("""## 2 · Direction

Does the sign of the forecast median call the sign of the actual move? Compare to
the naive 'always up' rule.""")
    code("""d = baseline.direction(j); d""")
    code("""print(f"up base rate this window:      {d['up_base_rate']:.1%}")
print(f"median-sign hit rate:          {d['median_sign_hit']:.1%}")
print(f"vs 'always guess the majority': {d['hit_vs_always_up']:+.1%}")
print("=> no directional edge at the 1-day horizon. Expected — daily equity "
      "direction is close to unpredictable from price alone.")""")

    md("## 3 · Bias & sharpness")
    code("""b = baseline.bias(j); b""")
    code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(j["q50"], j["ret"], s=6, alpha=.3)
lim = np.percentile(np.abs(j[["q50","ret"]].values), 99)
ax[0].plot([-lim,lim],[-lim,lim], c="k", lw=1)
ax[0].set_xlim(-lim,lim); ax[0].set_ylim(-lim,lim)
ax[0].set_xlabel("forecast median"); ax[0].set_ylabel("actual return"); ax[0].set_title("median vs actual")
ax[1].hist(j["signed_err"], bins=60); ax[1].axvline(0, c="k", lw=1)
ax[1].set_title(f"signed error (mean {b['mean_signed_err']:+.4f})")
plt.tight_layout(); plt.show()
print("median is ~unbiased; the v1 'mild positive lean' is gone at a 120-bar lookback.")""")

    md("""## 4 · Skill — does the model know which days matter?

This is the question that decides whether triage's ranking has anything to work
with. Two candidate signals: **strength** (|median| / dispersion) and raw
**dispersion**.""")
    code("""s = baseline.skill(j); s""")
    code("""fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, col, name in [(ax[0], "strength_abs", "|strength|"), (ax[1], "std", "dispersion")]:
    bins = pd.qcut(j[col], 10, duplicates="drop")
    m = j.groupby(bins, observed=True)["abs_ret"].mean()
    a.bar(range(len(m)), m.values); a.set_title(f"|actual ret| by {name} decile")
    a.set_xlabel(f"{name} decile (low→high)")
ax[0].set_ylabel("mean |actual return|")
plt.tight_layout(); plt.show()
print(f"Spearman |strength| vs |ret|:  {s['spearman_strength_vs_abs_ret']:+.3f}  (p={s['spearman_pvalue']})")
print(f"Spearman dispersion vs |ret|:  {s['spearman_dispersion_vs_abs_ret']:+.3f}")
print("=> raw strength barely orders eventful days; dispersion does better. "
      "Triage leans on strength_z (strength vs the name's own history) and the "
      "cluster read to compensate.")""")

    md("## 5 · Over time / by name")
    code("""baseline.by_week(j).round(3)""")
    code("""baseline.by_ticker(j).round(3).head(15)""")

    md("""---
## What this means for the project

Kronos-small on this task is an **unbiased but low-skill** forecaster: it calls
direction no better than chance and its intervals are ~12 points too tight. That
is not a criticism of the loop — it is the premise. The value the desk adds is
**disciplined review and accumulating memory around a mediocre model**, not
alpha. The eval (`eval/evaluate.py`) measures the loop against that honest bar:
does triage still surface the eventful names, does the post-mortem correctly
attribute the misses, does the memory's base rate predict later outcomes.""")
    return nb


def main() -> None:
    nb = build()
    out = __import__("pathlib").Path(__file__).parent / "kronos_baseline.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
