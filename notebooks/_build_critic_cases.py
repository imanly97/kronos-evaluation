"""Generate notebooks/critic_cases.ipynb — worked examples of the groundedness
critic: what it checks, what it catches, and a planted failure.

    .venv/bin/python notebooks/_build_critic_cases.py

Needs a watchlist with briefs (run triage on a few sessions first, e.g.
`python -m src.run_daily premarket --asof 2025-05-28 --skip-forecast`).
"""
from __future__ import annotations

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md("""# The groundedness critic — cases

Every number in every triage brief must trace to a value the deterministic core
computed. The critic runs two passes (`src/critic.py`):

1. **deterministic** — pull the numerals out of the prose, match each against the
   dossier's computed values within tolerance. Reproducible; this is the backbone
   of the "100% of numeric claims traceable" target.
2. **LLM** — read the brief against the same values and flag overclaims, a
   direction that contradicts the forecast, or a missing caveat.

Briefs loop draft → review until clean or `CRITIC_MAX_ROUNDS`.
""")

    code("""import os
if "src" not in os.listdir("."):
    os.chdir("..")""")
    code("""import pandas as pd
from src.store import load
from src.critic import review, check_numbers, CriticReport
w = load("watchlist")
print(len(w), "briefs on the watchlist across", w["asof"].nunique(), "sessions")
w[["asof","ticker","setup","groundedness","critic_rounds","critic_ok"]]""")

    md("## 1 · Groundedness across all briefs")
    code("""g = w["groundedness"].dropna()
print(f"mean groundedness at write time: {g.mean():.1%}")
print(f"briefs the critic passed clean:  {int(w['critic_ok'].fillna(False).sum())}/{len(w)}")
print(f"briefs needing >=1 revision:      {int((w['critic_rounds'].fillna(0) > 0).sum())}")
g.hist(bins=20); import matplotlib.pyplot as plt; plt.xlabel("groundedness"); plt.title("per-brief groundedness"); plt.show()""")

    md("""## 2 · The deterministic number-trace on one real brief

Every token the critic pulls, and whether it matched a computed value.""")
    code("""row = w.sort_values("groundedness").iloc[0]      # the weakest brief
print(f"{row['ticker']} — {row['asof']} — {row['setup']}\\n")
print(row["brief_md"])""")
    code("""# rebuild the value set the same way triage did
from src.features import rank_table
from src.forecasts import strength_history
from src.setups import setup_features
from src.data import load_context
f = load("forecasts"); asof, tk = str(row["asof"]), row["ticker"]
fr = f[(f["asof"].astype(str)==asof) & (f["ticker"]==tk)].iloc[0]
vals = {"q50":fr["q50"],"q05":fr["q05"],"q95":fr["q95"],"p_up":fr["p_up"],"std":fr["std"],"strength":fr["strength"]}
day = f[f["asof"].astype(str)==asof]
rt = rank_table(day[["ticker","q50","q05","q95","p_up","std","strength"]],
                strength_hist={t: strength_history(t, asof) for t in day["ticker"]})
vals["strength_z"] = float(rt[rt["ticker"]==tk].iloc[0]["strength_z"])
feats = setup_features(load_context(tk, asof).df, ticker=tk)
vals.update({k:v for k,v in feats.as_row().items() if isinstance(v,(int,float))})
if row["analog_base_rate"] is not None: vals["analog_base_rate"] = row["analog_base_rate"]
if row["analog_n"] is not None: vals["analog_n"] = row["analog_n"]

n, ok, bad = check_numbers(row["brief_md"], vals)
print(f"numbers found: {n}   grounded: {ok}   ungrounded: {len(bad)}")
for b in bad: print("  UNGROUNDED:", b)
pd.Series(vals).round(4)""")

    md("""## 3 · Planted failure — the critic must catch a wrong number

Take a clean brief, corrupt one figure, and confirm the critic flags it.""")
    code("""clean = w[w["critic_ok"].fillna(False)].iloc[0]
good_vals = {"q50":0.018,"p_up":0.82,"std":0.015,"strength":1.2,"analog_base_rate":0.63}
brief_ok = ("NVDA flagged on a momentum-breakout. Kronos median +1.8% with P(up) 82% "
            "and strength 1.2. Analogs: 63% up historically. "
            "Caveat: small analog sample, treat as descriptive.")
brief_bad = brief_ok.replace("+1.8%", "+7.4%").replace("82%", "97%")
for tag, b in [("clean", brief_ok), ("corrupted", brief_bad)]:
    r = review(b, good_vals, setup="momentum-breakout", require_caveat=True)
    print(f"[{tag}]  ok={r.ok}  groundedness={r.groundedness:.0%}  ungrounded={r.ungrounded}")""")

    md("""## 4 · The critic's feedback, verbatim

What gets sent back to the writer on a failed review.""")
    code("""r = review(brief_bad, good_vals, setup="momentum-breakout")
print(r.feedback())""")

    md("""---
The deterministic pass is the guarantee: a number that matches no computed value
cannot survive to the PM, regardless of what the LLM writer or LLM critic think.""")
    return nb


def main() -> None:
    nb = build()
    out = __import__("pathlib").Path(__file__).parent / "critic_cases.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
