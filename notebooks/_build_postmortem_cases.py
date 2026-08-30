"""Generate notebooks/postmortem_cases.ipynb — worked examples of the post-mortem
agent classifying *why* a forecast missed.

    .venv/bin/python notebooks/_build_postmortem_cases.py

Needs graded episodes (run `python -m src.run_daily postclose --asof <d>` on a
few sessions first).
"""
from __future__ import annotations

import nbformat as nbf


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))  # noqa: E731
    code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))    # noqa: E731

    md("""# The post-mortem agent — cases

For every watchlist name and every tail surprise (actual landed at
`actual_quantile` ≤ 0.10 or ≥ 0.90), the post-mortem classifies why
(`src/postmortem.py`):

| label | when |
|---|---|
| `within-expected-dispersion` | close landed mid-distribution — the flag was noise |
| `catalyst:earnings` | the name reported within ±1 day (curated calendar) |
| `catalyst:unlisted` | LLM identified a real company event the calendar missed |
| `regime-move` | the correlated cluster moved together |
| `data-issue` | a split / stale-price / thin-volume artefact |
| `unexplained` | tail move, idiosyncratic, no scheduled catalyst |

Deterministic guardrails decide the clear cases; the LLM only sees the
genuinely ambiguous idiosyncratic tails.
""")

    code("""import os
if "src" not in os.listdir("."):
    os.chdir("..")""")
    code("""import json, pandas as pd
from pathlib import Path
from src import config
eps = pd.DataFrame(json.loads(l) for l in
                   (config.MEMORY_DIR/"episodes.jsonl").read_text().splitlines() if l.strip())
print(len(eps), "episodes across", eps["asof"].nunique(), "sessions")
eps["classification"].value_counts()""")

    md("## 1 · One worked case per label")
    code("""from src.postmortem import sector_move, known_catalyst, data_sanity
def show(ep):
    t, a = ep["ticker"], ep["asof"]
    print(f"=== {t}  {a}  ({ep['setup']}) ===")
    print(f"forecast median {ep['forecast_median']:+.2%} | actual {ep['actual']:+.2%} "
          f"| landed at q{ep['actual_quantile']:.2f}")
    print(f"sector_move : {sector_move(t, a)}")
    print(f"catalyst    : {known_catalyst(t, a)}")
    print(f"data_sanity : {data_sanity(t, a)}")
    print(f"--> {ep['classification']}  —  {ep['note']}\\n")

for label in eps["classification"].unique():
    show(eps[eps["classification"] == label].iloc[0])""")

    md("""## 2 · Catalyst recall — the planted-label test (PLAN §7)

On sessions that coincide with a scheduled earnings date, does the post-mortem
name the catalyst?""")
    code("""from eval.evaluate import postmortem_catalyst_recall
pd.Series(postmortem_catalyst_recall())""")

    md("""## 3 · Dispersion self-consistency

When it says *within-expected-dispersion*, did the close actually land
mid-distribution? When it names an event, was the actual in the tail?""")
    code("""from eval.evaluate import postmortem_dispersion_consistency
pd.Series(postmortem_dispersion_consistency())""")

    md("## 4 · The memory this writes")
    code("""from src.memory import rebuild_stats, recall
stats = rebuild_stats()
print(len(stats), "(ticker, setup) keys in the derived stat table\\n")
for ep in eps.head(6).itertuples():
    print(recall(ep.ticker, ep.setup).line())""")

    md("""---
The `(ticker, setup)` stat lines above are exactly what the **next** morning's
triage `recall_memory` tool returns — the feedback edge that closes the loop.""")
    return nb


def main() -> None:
    nb = build()
    out = __import__("pathlib").Path(__file__).parent / "postmortem_cases.ipynb"
    nbf.write(nb, out)
    print("wrote", out)


if __name__ == "__main__":
    main()
