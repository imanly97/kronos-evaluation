"""Generate notebooks/walkthrough.ipynb — the loop end to end for one session.

    .venv/bin/python notebooks/_build_walkthrough.py --asof 2025-05-29

Keeps the notebook itself out of source control churn: the .py is the source,
the .ipynb is a build artifact you re-run.
"""
from __future__ import annotations

import argparse

import nbformat as nbf

ASOF_DEFAULT = "2025-05-29"


def build(asof: str) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md(f"""# Forecast Desk — one session, end to end

`asof = {asof}`. The deterministic forecast is assumed cached
(`python -m src.forecasts --asof {asof}`). This notebook walks the loop:

1. the no-lookahead price cut Kronos sees
2. the sampled path distribution (dispersion restored) + fan chart
3. cross-sectional ranking and the per-name strength z-score
4. the triage agent: setup → analogs → cluster → memory → keep/drop → brief
5. the groundedness critic
6. the close-of-day grade
7. the post-mortem: classify *why*, write a desk-memory episode
""")

    code("""import os
if "src" not in os.listdir("."):
    os.chdir("..")""")
    code("""import pandas as pd, matplotlib.pyplot as plt
from src import config
ASOF = %r
DEV = config.DEV_UNIVERSE
ASOF""" % asof)

    md("## 1 · The no-lookahead cut")
    code("""from src.data import load_context
pl = load_context("NVDA", ASOF)
print("source:", pl.source, "| context through:", pl.last_context_date.date(),
      "| bars:", len(pl.df))
assert pl.df.index.max() < pd.Timestamp(ASOF)   # strictly before asof
pl.df.tail()""")

    md("## 2 · Kronos with dispersion + fan chart")
    code("""from src.kronos_infer import forecast, fan_chart
fc = forecast(pl)
print(f"median {fc.median_ret*100:+.2f}%  P(up) {fc.p_up:.0%}  "
      f"dispersion {fc.std_ret*100:.2f}%  strength {fc.strength:+.2f}")
fan_chart(fc); plt.show()""")

    md("## 3 · Cross-sectional rank + strength-z")
    code("""from src.features import rank_table
from src.forecasts import strength_history
from src.store import load
f = load("forecasts"); f = f[f["asof"].astype(str) == ASOF]
hist = {t: strength_history(t, ASOF) for t in f["ticker"]}
rt = rank_table(f[["ticker","q50","q05","q95","p_up","std","strength"]], strength_hist=hist)
rt.head(12)""")

    md("## 4 · Triage agent")
    code("""from src.triage import run as run_triage
st = run_triage(ASOF)
print("\\n".join(st["trace"]))
print("\\nkept:", st["kept"])""")

    code("""print(open(config.LOG_DIR / f"brief_{ASOF}.md").read())""")

    md("## 5 · Groundedness critic (on one brief)")
    code("""from src.critic import review
from src.triage import _flat_values
t = st["kept"][0]
rep = review(st["briefs"][t]["brief_md"], _flat_values(st["dossiers"][t]),
             setup=st["dossiers"][t]["setup"])
print(f"{t}: groundedness {rep.groundedness:.0%}, ok={rep.ok}, "
      f"ungrounded={rep.ungrounded}, issues={rep.llm_issues}")""")

    md("## 6 · Close-of-day grade")
    code("""from src.grade import run as run_grade, report
graded = run_grade(ASOF)
pd.DataFrame(graded)[["ticker","ret","actual_quantile","inside_envelope","dir_match"]]""")

    md("## 7 · Post-mortem → desk memory")
    code("""from src.postmortem import run as run_pm
pm = run_pm(ASOF)
print("\\n".join(pm["trace"]))""")

    code("""from src.memory import recall
for t in st["kept"]:
    s = st["dossiers"][t]["setup"]
    print(recall(t, s).line())""")

    md("""---
The memory line above is what the **next** morning's triage reads back for these
names. Run `notebooks/loop_replay.ipynb` to watch it accumulate over the window.""")
    return nb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=ASOF_DEFAULT)
    args = ap.parse_args()
    nb = build(args.asof)
    out = __import__("pathlib").Path(__file__).parent / "walkthrough.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
