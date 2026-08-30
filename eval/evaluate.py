"""Evaluation harness (PLAN §7).

Every metric here is a check you can reason about — that is *why* this loop was
chosen over conformal-coverage math. Run what the store already supports:

    .venv/bin/python -m eval.evaluate                 # full report on current logs
    .venv/bin/python -m eval.evaluate --human 2025-05-29   # dump a day's raw table

Metrics
  triage.groundedness       % of numeric claims in briefs tied to a computed value
  triage.eventfulness       flagged vs unflagged: |ret|, realised range, tail-ness
  triage.human_agreement    overlap with your own top-5 picks (needs eval/human_picks.json)
  postmortem.catalyst_recall on scheduled-earnings days, did it say catalyst:earnings?
  postmortem.dispersion_consistency  labels vs where the actual actually landed
  memory.base_rate_validity does a setup's claimed hit-rate predict later outcomes?

`--ablation` compares two runs logged under different LOG_DIRs (memory on vs off).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import POSTMORTEM_TAIL_Q, ROOT
from src.memory import _load_episodes
from src.store import joined, load

HUMAN_PICKS = ROOT / "eval" / "human_picks.json"
CATALYSTS = ROOT / "eval" / "catalysts.json"


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------- #
# triage
# --------------------------------------------------------------------------- #
def triage_groundedness() -> dict:
    """Two readings: the critic's own verdict at write time (it had the full
    dossier), and an independent re-check that rebuilds the value set here."""
    w = load("watchlist")
    if w.empty or "brief_md" not in w.columns:
        return {"n_briefs": 0}
    from src.critic import check_numbers
    from src.features import rank_table
    from src.forecasts import strength_history
    from src.setups import setup_features
    from src.data import load_context

    f = load("forecasts")
    tot = ok = 0
    per = []
    for _, r in w.iterrows():
        asof, tk = str(r["asof"]), r["ticker"]
        vals: dict[str, float] = {}
        frow = f[(f["asof"].astype(str) == asof) & (f["ticker"] == tk)]
        if not frow.empty:
            fr = frow.iloc[0]
            vals.update({"q50": fr["q50"], "q05": fr["q05"], "q95": fr["q95"],
                         "p_up": fr["p_up"], "std": fr["std"], "strength": fr["strength"]})
            day_f = f[f["asof"].astype(str) == asof]
            hist = {t: strength_history(t, asof) for t in day_f["ticker"]}
            rt = rank_table(day_f[["ticker", "q50", "q05", "q95", "p_up", "std", "strength"]],
                            strength_hist=hist)
            rr = rt[rt["ticker"] == tk]
            if not rr.empty:
                vals["strength_z"] = float(rr.iloc[0]["strength_z"])
        try:
            feats = setup_features(load_context(tk, asof).df, ticker=tk)
            vals.update({k: v for k, v in feats.as_row().items() if isinstance(v, (int, float))})
        except Exception:  # noqa: BLE001
            pass
        if r.get("analog_base_rate") is not None:
            vals["analog_base_rate"] = r["analog_base_rate"]
        if r.get("analog_n") is not None:
            vals["analog_n"] = r["analog_n"]
        n, g, _ = check_numbers(r["brief_md"], vals)
        tot += n
        ok += g
        per.append(g / n if n else 1.0)

    out = {
        "n_briefs": len(w), "n_numbers": tot, "n_grounded": ok,
        "groundedness_recheck": round(ok / tot, 3) if tot else 1.0,
        "worst_brief_recheck": round(min(per), 3) if per else 1.0,
    }
    if "groundedness" in w.columns and w["groundedness"].notna().any():
        gg = w["groundedness"].dropna()
        out["groundedness_at_write"] = round(float(gg.mean()), 3)
        out["briefs_critic_clean"] = int(w["critic_ok"].fillna(False).sum())
    return out


def triage_eventfulness() -> dict:
    j = joined()
    if j.empty or "ret" not in j.columns:
        return {"graded": 0}
    j = j.dropna(subset=["ret"])
    if j.empty or "flagged" not in j.columns:
        return {"graded": 0}
    fl, un = j[j["flagged"]], j[~j["flagged"]]
    if fl.empty or un.empty:
        return {"graded": len(j), "flagged": len(fl), "note": "need both groups"}

    def _tail(g):
        return (g["actual_quantile"] - 0.5).abs().mean()

    return {
        "graded": len(j), "flagged": len(fl), "unflagged": len(un),
        "abs_ret_flagged": round(fl["ret"].abs().mean(), 4),
        "abs_ret_unflagged": round(un["ret"].abs().mean(), 4),
        "abs_ret_ratio": round(fl["ret"].abs().mean() / max(un["ret"].abs().mean(), 1e-9), 2),
        "tailness_flagged": round(_tail(fl), 3),
        "tailness_unflagged": round(_tail(un), 3),
        "envelope_break_flagged": round(1 - fl["inside_envelope"].mean(), 3),
        "envelope_break_unflagged": round(1 - un["inside_envelope"].mean(), 3),
    }


def triage_human_agreement() -> dict:
    if not HUMAN_PICKS.exists():
        return {"note": f"no {HUMAN_PICKS.name}; run `--human <date>` on ~20 days and fill it in"}
    picks = json.loads(HUMAN_PICKS.read_text())
    w = load("watchlist")
    overlaps = []
    for asof, mine in picks.items():
        agent = set(w[w["asof"].astype(str) == str(asof)]["ticker"])
        if not agent:
            continue
        mine = set(mine)
        overlaps.append(len(mine & agent) / max(len(mine), 1))
    if not overlaps:
        return {"note": "no overlapping dates between picks and watchlist"}
    return {"days": len(overlaps), "mean_overlap": round(float(np.mean(overlaps)), 3)}


# --------------------------------------------------------------------------- #
# post-mortem
# --------------------------------------------------------------------------- #
def postmortem_catalyst_recall() -> dict:
    eps = _load_episodes()
    if not eps or not CATALYSTS.exists():
        return {"n": 0}
    cat = json.loads(CATALYSTS.read_text())
    earn = {t: set(v) for t, v in cat.get("earnings", {}).items()}

    def _near(ticker, asof):
        from datetime import date
        d = date.fromisoformat(str(asof))
        return any(abs((date.fromisoformat(x) - d).days) <= 1 for x in earn.get(ticker, ()))

    on_earn = [e for e in eps if _near(e["ticker"], e["asof"])]
    if not on_earn:
        return {"n": 0, "note": "no episodes coincide with a scheduled earnings date"}
    hit = sum(str(e["classification"]).startswith("catalyst") for e in on_earn)
    lo, hi = _wilson(hit, len(on_earn))
    return {"n": len(on_earn), "named_catalyst": hit,
            "recall": round(hit / len(on_earn), 3), "wilson95": [round(lo, 2), round(hi, 2)]}


def postmortem_dispersion_consistency() -> dict:
    eps = _load_episodes()
    if not eps:
        return {"n": 0}
    within = [e for e in eps if e["classification"] == "within-expected-dispersion"]
    something = [e for e in eps if e["classification"] in ("regime-move",)
                 or str(e["classification"]).startswith("catalyst")]
    out = {"n": len(eps)}
    if within:
        q = np.array([e["actual_quantile"] for e in within])
        out["within_label_n"] = len(within)
        out["within_label_median_|q-.5|"] = round(float(np.median(np.abs(q - 0.5))), 3)
        out["within_label_off_tail"] = round(
            float(np.mean((q > POSTMORTEM_TAIL_Q) & (q < 1 - POSTMORTEM_TAIL_Q))), 3)
    if something:
        q = np.array([e["actual_quantile"] for e in something])
        out["event_label_n"] = len(something)
        out["event_label_in_tail"] = round(
            float(np.mean((q <= POSTMORTEM_TAIL_Q) | (q >= 1 - POSTMORTEM_TAIL_Q))), 3)
    return out


# --------------------------------------------------------------------------- #
# memory
# --------------------------------------------------------------------------- #
def memory_base_rate_validity(min_n: int = 3) -> dict:
    """For each (ticker, setup), does the hit-rate over the FIRST half of its
    episodes predict direction accuracy over the SECOND half?"""
    eps = _load_episodes()
    by: dict[str, list] = {}
    for e in sorted(eps, key=lambda x: str(x["asof"])):
        by.setdefault(f"{e['ticker']}|{e['setup']}", []).append(e)

    rows = []
    for key, seq in by.items():
        if len(seq) < 2 * min_n:
            continue
        mid = len(seq) // 2
        early, late = seq[:mid], seq[mid:]

        def _hit(g):
            h = [1 for e in g if e["forecast_median"] is not None and e["actual"] is not None
                 and (e["forecast_median"] > 0) == (e["actual"] > 0) and e["actual"] != 0]
            return len(h) / len(g)

        rows.append((key, _hit(early), _hit(late)))
    if len(rows) < 3:
        return {"pairs": len(rows), "note": f"need >= {2*min_n} episodes for >= 3 keys"}
    e_arr = np.array([r[1] for r in rows])
    l_arr = np.array([r[2] for r in rows])
    return {
        "pairs": len(rows),
        "corr_early_vs_late_hit": round(float(np.corrcoef(e_arr, l_arr)[0, 1]), 3),
        "mean_abs_drift": round(float(np.mean(np.abs(e_arr - l_arr))), 3),
    }


# --------------------------------------------------------------------------- #
# report / CLI
# --------------------------------------------------------------------------- #
def _dump_human_table(asof: str) -> None:
    from src.features import rank_table
    from src.forecasts import strength_history

    f = load("forecasts")
    f = f[f["asof"].astype(str) == str(asof)]
    if f.empty:
        print(f"no forecasts cached for {asof}")
        return
    hist = {t: strength_history(t, asof) for t in f["ticker"]}
    rt = rank_table(f[["ticker", "q50", "q05", "q95", "p_up", "std", "strength"]], strength_hist=hist)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(rt.head(20).to_string(index=False))
    print(f'\nAdd your top-5 to {HUMAN_PICKS.name} as  {{"{asof}": ["T1","T2","T3","T4","T5"]}}')


def report() -> str:
    blocks = {
        "triage.groundedness": triage_groundedness(),
        "triage.eventfulness": triage_eventfulness(),
        "triage.human_agreement": triage_human_agreement(),
        "postmortem.catalyst_recall": postmortem_catalyst_recall(),
        "postmortem.dispersion_consistency": postmortem_dispersion_consistency(),
        "memory.base_rate_validity": memory_base_rate_validity(),
    }
    out = ["=" * 64, "forecast-desk evaluation", "=" * 64]
    for name, res in blocks.items():
        out.append(f"\n[{name}]")
        for k, v in res.items():
            out.append(f"  {k:32} {v}")
    return "\n".join(out)


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Forecast-desk evaluation harness.")
    ap.add_argument("--human", metavar="ASOF", help="dump a day's raw rank table for manual picks")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    if args.human:
        _dump_human_table(args.human)
        return 0
    if args.json:
        print(json.dumps({
            "triage_groundedness": triage_groundedness(),
            "triage_eventfulness": triage_eventfulness(),
            "postmortem_catalyst_recall": postmortem_catalyst_recall(),
            "postmortem_dispersion_consistency": postmortem_dispersion_consistency(),
            "memory_base_rate_validity": memory_base_rate_validity(),
        }, indent=2, default=str))
        return 0
    print(report())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
