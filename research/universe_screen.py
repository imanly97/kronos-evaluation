"""Pick the ~30-name evaluation universe with a liquidity screen (PLAN §4).

Criteria:
  - daily dollar volume (close x shares): median >= $300M over the trailing year,
    and >= $50M on the *worst* day (no illiquid air pockets)
  - continuous daily history back to 2018-01
  - keep the most-liquid names, balanced across GICS-ish sectors

Self-contained fetch (curl_cffi -> Yahoo chart API). Writes universe_screen.csv
and prints the proposed list for src/config.py.

    .venv/bin/python research/universe_screen.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from curl_cffi import requests as cr

OUT = Path(__file__).parent / "universe_screen.csv"
MIN_MED_ADV = 300e6
MIN_WORST_ADV = 50e6
NEED_HISTORY_FROM = pd.Timestamp("2018-01-01")
PER_SECTOR_MAX = 5
TARGET_N = 30

# candidate pool — large, liquid US names across sectors
POOL: dict[str, str] = {
    # Information Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech", "ORCL": "Tech",
    "CRM": "Tech", "AMD": "Tech", "ADBE": "Tech", "CSCO": "Tech", "ACN": "Tech",
    "INTC": "Tech", "QCOM": "Tech", "TXN": "Tech", "IBM": "Tech", "NOW": "Tech",
    "INTU": "Tech", "MU": "Tech", "AMAT": "Tech",
    # Communication Services
    "GOOGL": "Comm", "META": "Comm", "NFLX": "Comm", "DIS": "Comm", "CMCSA": "Comm",
    "T": "Comm", "VZ": "Comm", "TMUS": "Comm",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc", "MCD": "ConsDisc",
    "NKE": "ConsDisc", "LOW": "ConsDisc", "SBUX": "ConsDisc", "BKNG": "ConsDisc",
    "TJX": "ConsDisc",
    # Consumer Staples
    "WMT": "ConsStap", "PG": "ConsStap", "KO": "ConsStap", "PEP": "ConsStap",
    "COST": "ConsStap", "PM": "ConsStap", "MDLZ": "ConsStap",
    # Financials
    "JPM": "Fin", "BAC": "Fin", "WFC": "Fin", "GS": "Fin", "MS": "Fin", "V": "Fin",
    "MA": "Fin", "BLK": "Fin", "SPGI": "Fin", "AXP": "Fin", "C": "Fin", "SCHW": "Fin",
    # Health Care
    "UNH": "Health", "JNJ": "Health", "LLY": "Health", "MRK": "Health", "ABBV": "Health",
    "PFE": "Health", "TMO": "Health", "ABT": "Health", "DHR": "Health", "AMGN": "Health",
    "BMY": "Health",
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    # Industrials
    "CAT": "Indu", "BA": "Indu", "HON": "Indu", "GE": "Indu", "UPS": "Indu",
    "RTX": "Indu", "UNP": "Indu", "DE": "Indu",
    # Materials / Utilities / Real Estate
    "LIN": "Materials", "SHW": "Materials",
    "NEE": "Utilities", "SO": "Utilities", "DUK": "Utilities",
    "AMT": "RealEstate", "PLD": "RealEstate",
}


def fetch_daily(ticker: str) -> pd.DataFrame | None:
    url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
           "?range=10y&interval=1d&events=div%2Csplit")
    for attempt in range(3):
        try:
            r = cr.get(url, impersonate="chrome124", timeout=30)
            r.raise_for_status()
            res = r.json()["chart"]["result"][0]
            ts = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(None).normalize()
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({"close": q["close"], "volume": q["volume"]}, index=ts).dropna()
            return df
        except Exception as exc:  # noqa: BLE001
            if attempt == 2:
                print(f"  {ticker}: fetch failed ({exc})", file=sys.stderr)
                return None
            time.sleep(2)
    return None


def main() -> None:
    rows = []
    for tk, sector in POOL.items():
        df = fetch_daily(tk)
        if df is None or df.empty:
            continue
        dv = (df["close"] * df["volume"]).dropna()
        last_yr = dv[dv.index >= dv.index.max() - pd.Timedelta(days=365)]
        rows.append({
            "ticker": tk, "sector": sector,
            "history_from": df.index.min().date(),
            "med_adv_$m": last_yr.median() / 1e6,
            "worst_adv_$m": last_yr.min() / 1e6,
            "last_close": float(df["close"].iloc[-1]),
        })
        time.sleep(0.15)

    d = pd.DataFrame(rows)
    d["has_history"] = pd.to_datetime(d["history_from"]) <= NEED_HISTORY_FROM
    d["passes"] = (d["med_adv_$m"] >= MIN_MED_ADV / 1e6) & \
                  (d["worst_adv_$m"] >= MIN_WORST_ADV / 1e6) & d["has_history"]
    d = d.sort_values("med_adv_$m", ascending=False)
    d.to_csv(OUT, index=False)

    # sector-balanced pick from the passing set
    picked, per_sector = [], {}
    for _, r in d[d["passes"]].iterrows():
        if per_sector.get(r["sector"], 0) >= PER_SECTOR_MAX:
            continue
        picked.append(r["ticker"]); per_sector[r["sector"]] = per_sector.get(r["sector"], 0) + 1
        if len(picked) >= TARGET_N:
            break

    print(d[["ticker", "sector", "history_from", "med_adv_$m", "worst_adv_$m", "passes"]]
          .to_string(index=False, float_format=lambda x: f"{x:,.0f}"))
    print(f"\n{d['passes'].sum()} / {len(d)} names pass the screen")
    print(f"\nsector spread of the pick: {per_sector}")
    print(f"\nUNIVERSE ({len(picked)}):")
    print("    " + ", ".join(f'"{t}"' for t in sorted(picked)))
    print(f"\nfull table -> {OUT}")


if __name__ == "__main__":
    main()
