"""Builds a per-case-study notebook: notebooks/case_<slug>.ipynb

Usage:
    python notebooks/_build_case.py nflx_2024-04-19
    python notebooks/_build_case.py jpm_2025-06-11
    python notebooks/_build_case.py all

Each case's prose lives in CASES below; the pipeline cells are shared.
"""
import json
import sys
from pathlib import Path

CASES = {
    "nflx_2024-04-19": dict(
        ticker="NFLX", asof="2024-04-19", kind="narrative head-fake",
        title="Case Study 03 — NFLX, Q1 2024 earnings (grade day 2024-04-19)",
        why=r"""
The overnight headlines were **unambiguously strong** — Netflix added 9.3M
subscribers against a ~5M estimate, beat on revenue and EPS, and expanded margins.
A naive read of the news flow says *bull, high conviction*.

The stock fell **9%**.

Two things in the same release did the damage: Netflix announced it would **stop
reporting quarterly subscriber numbers** from 2025 (the metric the whole bull
case had rested on), and guided Q2 revenue slightly light. This is the case where
narrative is confidently, expensively wrong.

And Kronos — which saw only prices through the 2024-04-18 close, on a stock that
had been grinding higher — *also* leaned bullish. So this day is a **double
miss**: both signals pointed up, the stock gapped down hard, and the realised
move fell straight through the bottom of Kronos's 5–95 envelope. It's the case
that tests coverage, not divergence.
""",
        reading=r"""
### Reading this case

NFLX closed **−9.1%** (grade cell above). Both signals missed it:

- **Narrative** returned *bull, conviction 7*. The scorer's own rationale names
  the "stop reporting subscriber counts from 2025" line and the light Q2 guide as
  offsets — and then concludes "the magnitude of the beat should dominate." It
  reasoned about the exact factor that sank the stock and still called it wrong.
  This is what a narrative head-fake looks like: the news genuinely was good, the
  market had already bought it, and the fine print mattered more than the beat.
- **Structure** leaned *bull, +1.1% median*, because NFLX had been trending up
  and Kronos extrapolates. The actual −9.1% landed **below the 5th percentile** of
  its envelope — a clean coverage failure.

No divergence here (both bullish), so no divergence-outcome row — but the
direction-hit and envelope-coverage stats both take the hit. The pilot lesson:
raw narrative conviction has to be discounted when the news is loud and
one-sided, and Kronos's envelope is not wide enough to cover a gap-down driven by
information it never saw.
""",
    ),
    "jpm_2025-06-11": dict(
        ticker="JPM", asof="2025-06-11", kind="no-catalyst day",
        title="Case Study 02 — JPM, a quiet mid-June session (grade day 2025-06-11)",
        why=r"""
The counterpoint to both earnings cases: **a day with no material news**. JPMorgan
doesn't report until mid-July; the overnight coverage is routine analyst notes and
a generic "banks await CPI" market wrap. There is no catalyst for narrative to
read.

The question this case exists to answer: **when narrative correctly abstains, does
the structural signal alone carry any information — or is a no-catalyst day just
noise for both sides?** If narrative adds value only by staying quiet here, that's
still a finding.
""",
        reading=r"""
### Reading this case

JPM barely moved (see the grade cell). With no catalyst, the headline sub-agent
should have filtered the routine coverage down to nothing material and the
narrative score should be neutral / low-conviction — narrative correctly abstains.

That makes the row a test of **Kronos's calibration with narrative silent**: does
the actual close land inside the 5–95 envelope, and is the envelope an honest
width for a day the model has no particular view? A near-zero realised move
sitting comfortably mid-envelope is the expected, boring, *good* outcome.

Part 2 reports envelope coverage separately for catalyst and no-catalyst days —
if the model is only well-calibrated on quiet days, that itself is worth knowing.
""",
    ),
}

C = []
def md(s): C.append(("markdown", s.strip("\n")))
def code(s): C.append(("code", s.strip("\n")))


def build(slug: str):
    global C
    C = []
    cfg = CASES[slug]
    md(f"# {cfg['title']}\n\n**Type:** {cfg['kind']}  ·  **Ticker:** {cfg['ticker']}  "
       f"·  **asof:** {cfg['asof']}\n\n"
       "Runs the full pipeline (`src/graph.py`) for this one signal-day and shows "
       "every stage. Companion to [`notebooks/walkthrough.ipynb`](walkthrough.ipynb), "
       "which explains the machinery in detail.")
    md("## Why this case\n" + cfg["why"])

    code(f"""
import sys, os, json, shutil, pathlib
sys.path.insert(0, "..")
DEMO_LOG = pathlib.Path("_out/case_{slug}").resolve()
if DEMO_LOG.exists(): shutil.rmtree(DEMO_LOG)
DEMO_LOG.mkdir(parents=True)
os.environ["NARB_LOG_DIR"] = str(DEMO_LOG)

import numpy as np, pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 100)
from src.llm import have_key
TICKER, ASOF = "{cfg['ticker']}", "{cfg['asof']}"
print("real LLM:", have_key(), " (mock mode if False)")
""")

    md("## 1 · Data — the no-lookahead cut")
    code("""
from src.data import load_context, load_actual
pl = load_context(TICKER, ASOF)
print(f"source {pl.source} · {len(pl.df)} bars · {pl.df.index.min().date()} -> {pl.last_context_date.date()}")
assert pl.last_context_date < pd.Timestamp(ASOF)
ax = pl.df["close"].iloc[-90:].plot(figsize=(10,3.6), color="#222", lw=1.2)
ax.axvline(pd.Timestamp(ASOF), color="#e45756", ls="--", label=f"asof {ASOF} (unseen)")
ax.set_title(f"{TICKER} close — Kronos context ends at the dashed line"); ax.legend(); plt.tight_layout()
""")

    md("## 2 · Structure — Kronos distribution")
    code("""
from src.kronos_infer import forecast, fan_chart
fc = forecast(pl)
q = fc.ret_quantiles()
print(f"median {fc.median_ret*100:+.2f}%  P(up) {fc.p_up:.0%}  dispersion {fc.std_ret*100:.2f}%  strength {fc.strength:+.2f}")
print("5-95 envelope:", f"[{q[0.05]*100:+.2f}%, {q[0.95]*100:+.2f}%]")
_ = fan_chart(fc, context_bars=45)
""")

    md("## 3 · Headlines sub-agent (price-blind)")
    code("""
from src.headlines_agent import get_headlines
hs = get_headlines(TICKER, ASOF)
print("mode:", hs["mode"], "| no_news:", hs["no_news"])
for n in hs["notes"]: print("  -", n)
pd.DataFrame(hs["headlines"])[["source","published","title"]] if hs["headlines"] else "— nothing material —"
""")

    md("## 4 · Narrative score")
    code("""
from src.narrative import score_narrative
sc = score_narrative(TICKER, hs["headlines"], ASOF)
print(json.dumps(sc.to_dict(), indent=2))
""")

    md("## 5 · Divergence verdict")
    code("""
from src.divergence import assess
div = assess(sc, fc, brief=True)
print(f"structure dir {div.struct_dir:+d} ({div.struct_median*100:+.2f}%)  |  "
      f"narrative dir {div.narr_dir:+d} (conv {sc.conviction})")
print(f"P(move in narrative's direction | structure) = {div.p_narr_direction:.0%}")
print(f"\\n==> {div.divergence_type}\\n\\n{div.brief}")
""")

    md("## 6 · Lock + grade at the close")
    code("""
from src.graph import run_ticker, _summary_row
from src.evaluate import run as run_eval
from src.store import joined
run_ticker(TICKER, ASOF, allow_replace=True)
run_eval(ASOF)
row = joined().query("ticker==@TICKER and asof==@ASOF").iloc[0]
a = load_actual(TICKER, ASOF)
print(f"realised close-to-close: {a.ret*100:+.2f}%")
row[["ret","kronos_q05","kronos_q50","kronos_q95","inside_envelope",
     "dir_match_struct","dir_match_narr","divergence_type","divergence_outcome"]]
""")
    code("""
_ = fan_chart(fc, actual_close=a.actual_close, context_bars=45)
plt.gca().set_title(plt.gca().get_title() +
    f"\\nactual {a.ret*100:+.2f}%  ->  {'inside' if row['inside_envelope'] else 'OUTSIDE'} the 5-95 envelope")
""")

    md("## 7 · Reading\n" + cfg["reading"]
       + "\n\n*(the numbers above are filled from this run; the narrative "
         "take assumes the real LLM scorer — re-run with `ANTHROPIC_API_KEY` set "
         "if you see `[MOCK]`.)*")

    nb = {"cells": [{"cell_type": k, "metadata": {}, "id": f"c{i:02d}",
                     "source": s.splitlines(keepends=True),
                     **({"outputs": [], "execution_count": None} if k == "code" else {})}
                    for i, (k, s) in enumerate(C)],
          "metadata": {"kernelspec": {"display_name": "narrative-arb (.venv)",
                                      "language": "python", "name": "narrative-arb"},
                       "language_info": {"name": "python", "version": "3.11"}},
          "nbformat": 4, "nbformat_minor": 5}
    out = Path(__file__).with_name(f"case_{slug}.ipynb")
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out.name, "-", len(C), "cells")


if __name__ == "__main__":
    args = sys.argv[1:] or ["all"]
    todo = list(CASES) if args == ["all"] else args
    for s in todo:
        build(s)
