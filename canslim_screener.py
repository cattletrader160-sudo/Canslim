#!/usr/bin/env python3
"""
CAN SLIM stock screener (William O'Neil method)
===============================================

Scores ASX and/or US stocks against the seven CAN SLIM letters and prints
the top 10. Results are also saved to a CSV file.

Setup (once):
    pip install yfinance pandas requests lxml

Run:
    python canslim_screener.py                  # ASX 200 (default)
    python canslim_screener.py --market us      # S&P 500
    python canslim_screener.py --market both    # ASX 200 + S&P 500
    python canslim_screener.py --tickers-file my_list.txt   # your own list
    python canslim_screener.py --top 20
    python canslim_screener.py --market both --json results.json   # for the app

How each letter is tested (free Yahoo Finance data, so these are proxies):
    C  Latest quarter's EPS up >= 25% on the same quarter a year ago
       (ASX firms report half-yearly, so Yahoo's quarterly figure may be used)
    A  Annual EPS grew >= 25% a year (compound) over the last 3+ years,
       with no down years
    N  Price within 15% of its 52-week high and above its 50-day average
       (a stand-in for "new highs"; new products/management aren't in the data)
    S  More volume on up days than down days over the last 50 days
       (accumulation), with reasonable liquidity
    L  Relative strength rating >= 80, i.e. outperformed 80% of the stocks
       screened over 12 months, weighted towards the latest quarter
    I  At least 20% of shares held by institutions (if Yahoo has the figure)
    M  Market index above its 50- and 200-day averages = uptrend

Not financial advice. Data can be missing or wrong; check before acting.
"""

import argparse
import io
import json
import math
import os
import sys
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    sys.exit("yfinance is not installed. Run:  pip install yfinance pandas requests lxml")

try:
    import requests
except ImportError:
    requests = None

# ----------------------------------------------------------------------------
# Thresholds (edit these to tune the screen)
# ----------------------------------------------------------------------------
C_MIN_QTR_EPS_GROWTH = 0.25      # 25%
A_MIN_ANNUAL_EPS_CAGR = 0.25     # 25% a year
N_MAX_BELOW_HIGH = 0.15          # within 15% of 52-week high
S_MIN_UPDOWN_VOL_RATIO = 1.0     # up-day volume / down-day volume
S_MIN_AVG_DOLLAR_VOL = 1_000_000 # 50-day average daily turnover (local currency)
L_MIN_RS_RATING = 80             # 1-99 scale
I_MIN_INST_HOLDING = 0.20        # 20%

PRICE_HISTORY = "15mo"
DOWNLOAD_CHUNK = 100             # tickers per price download batch

INDEXES = {"asx": ("^AXJO", "S&P/ASX 200"), "us": ("^GSPC", "S&P 500")}

# Fallback lists if Wikipedia can't be reached
FALLBACK_ASX = [
    "BHP", "CBA", "CSL", "NAB", "WBC", "ANZ", "WES", "MQG", "GMG", "FMG",
    "TLS", "RIO", "WOW", "WDS", "TCL", "ALL", "QBE", "REA", "STO", "COL",
    "XRO", "PME", "WTC", "SUN", "JHX", "RMD", "CPU", "NST", "EVN", "TNE",
    "CAR", "SEK", "HUB", "NWL", "QAN", "BXB", "ORG", "IAG", "AMC", "S32",
    "LYC", "PLS", "MIN", "ALQ", "SHL", "COH", "TPG", "NXT", "CWY", "JBH",
]
FALLBACK_US = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "BRK-B",
    "LLY", "JPM", "V", "UNH", "XOM", "MA", "COST", "HD", "PG", "NFLX", "JNJ",
    "ABBV", "CRM", "AMD", "ORCL", "ADBE", "BAC", "KO", "PEP", "TMO", "MRK",
    "CSCO", "ACN", "LIN", "MCD", "WMT", "INTU", "NOW", "ISRG", "UBER", "PLTR",
    "ANET", "CRWD", "PANW", "BKNG", "AXP", "GE", "CAT", "DE", "LRCX", "KLAC",
]


# ----------------------------------------------------------------------------
# Universe
# ----------------------------------------------------------------------------
def _read_wiki_tables(url):
    if requests is None:
        raise RuntimeError("requests not installed")
    headers = {"User-Agent": "Mozilla/5.0 (CAN SLIM screener)"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return pd.read_html(io.StringIO(resp.text))


def get_asx_universe():
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/S%26P/ASX_200")
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            for key in ("code", "ticker", "asx code", "symbol"):
                if key in cols:
                    codes = t.iloc[:, cols.index(key)].astype(str).str.strip()
                    codes = [c for c in codes if c.isalnum() and 2 <= len(c) <= 4]
                    if len(codes) > 100:
                        return [c.upper() + ".AX" for c in codes]
        raise RuntimeError("table not found")
    except Exception as e:
        print(f"  (Couldn't fetch ASX 200 list: {e}. Using built-in list of 50.)")
        return [c + ".AX" for c in FALLBACK_ASX]


def get_us_universe():
    try:
        tables = _read_wiki_tables("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        syms = tables[0]["Symbol"].astype(str).str.strip().str.replace(".", "-", regex=False)
        syms = list(syms)
        if len(syms) > 400:
            return syms
        raise RuntimeError("table too short")
    except Exception as e:
        print(f"  (Couldn't fetch S&P 500 list: {e}. Using built-in list of 50.)")
        return list(FALLBACK_US)


def load_tickers_file(path):
    with open(path) as f:
        raw = f.read().replace(",", "\n").split()
    return [t.strip().upper() for t in raw if t.strip() and not t.startswith("#")]


# ----------------------------------------------------------------------------
# Price data
# ----------------------------------------------------------------------------
def download_prices(tickers):
    """Return {ticker: DataFrame with Open/High/Low/Close/Volume}."""
    out = {}
    for i in range(0, len(tickers), DOWNLOAD_CHUNK):
        chunk = tickers[i:i + DOWNLOAD_CHUNK]
        print(f"  Downloading prices {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        try:
            data = yf.download(chunk, period=PRICE_HISTORY, interval="1d",
                               auto_adjust=True, group_by="ticker",
                               threads=True, progress=False)
        except Exception as e:
            print(f"    download failed: {e}")
            continue
        if data is None or data.empty:
            continue
        if isinstance(data.columns, pd.MultiIndex):
            level0 = set(data.columns.get_level_values(0))
            for t in chunk:
                if t in level0:
                    df = data[t].dropna(how="all")
                    if not df.empty:
                        out[t] = df
        elif len(chunk) == 1:
            out[chunk[0]] = data.dropna(how="all")
    return out


def market_direction(index_symbol):
    try:
        df = yf.download(index_symbol, period="15mo", interval="1d",
                         auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        last = float(close.iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1])
        ma200 = float(close.rolling(200).mean().iloc[-1])
    except Exception as e:
        return "Unknown", f"couldn't load index ({e})"
    if last > ma50 > ma200:
        status = "Uptrend"
    elif last > ma200:
        status = "Mixed"
    else:
        status = "Downtrend"
    detail = f"{last:,.0f} vs 50-day {ma50:,.0f} / 200-day {ma200:,.0f}"
    return status, detail


def price_metrics(df):
    """Compute N, S and raw relative-strength inputs from price history."""
    close = df["Close"].astype(float).dropna()
    vol = df["Volume"].astype(float).reindex(close.index).fillna(0)
    if len(close) < 200:
        return None

    last = close.iloc[-1]
    high52 = df["High"].astype(float).iloc[-252:].max()
    ma50 = close.rolling(50).mean().iloc[-1]
    pct_below_high = 1 - last / high52 if high52 > 0 else np.nan

    def ret(n):
        n = min(n, len(close) - 1)
        return close.iloc[-1] / close.iloc[-1 - n] - 1

    # O'Neil-style weighting: latest quarter counts double
    rs_raw = 2 * ret(63) + ret(126) + ret(189) + ret(252)

    chg = close.diff().iloc[-50:]
    v50 = vol.iloc[-50:]
    up_vol = v50[chg > 0].sum()
    down_vol = v50[chg < 0].sum()
    updown = up_vol / down_vol if down_vol > 0 else np.inf
    avg_dollar_vol = (close.iloc[-50:] * v50).mean()

    return {
        "Price": round(float(last), 2),
        "% Below High": round(float(pct_below_high) * 100, 1),
        "rs_raw": float(rs_raw),
        "12m Return %": round(float(ret(252)) * 100, 1),
        "Up/Down Vol": round(float(updown), 2) if np.isfinite(updown) else 99.0,
        "avg_dollar_vol": float(avg_dollar_vol),
        "N": bool(pct_below_high <= N_MAX_BELOW_HIGH and last > ma50),
        "S": bool(updown >= S_MIN_UPDOWN_VOL_RATIO and avg_dollar_vol >= S_MIN_AVG_DOLLAR_VOL),
    }


# ----------------------------------------------------------------------------
# Fundamentals
# ----------------------------------------------------------------------------
EPS_ROWS = ["Diluted EPS", "Basic EPS"]
NI_ROWS = ["Net Income Common Stockholders", "Net Income",
           "Net Income From Continuing Operation Net Minority Interest"]


def _series_from_statement(stmt):
    if stmt is None or getattr(stmt, "empty", True):
        return None
    for row in EPS_ROWS + NI_ROWS:
        if row in stmt.index:
            s = pd.to_numeric(stmt.loc[row], errors="coerce").dropna()
            if len(s) >= 2:
                return s.sort_index(ascending=False)  # newest first
    return None


def _growth(new, old):
    if old is None or new is None or pd.isna(old) or pd.isna(new) or old <= 0:
        return None  # growth from a loss or zero isn't meaningful
    return new / old - 1


def fundamentals(ticker):
    t = yf.Ticker(ticker)
    res = {"Qtr EPS Growth %": None, "Annual EPS CAGR %": None,
           "Inst. Held %": None, "ROE %": None, "Name": ticker}

    info = {}
    try:
        info = t.info or {}
    except Exception:
        pass
    res["Name"] = (info.get("shortName") or ticker)[:28]

    # C: current quarterly earnings
    qg = None
    try:
        q = _series_from_statement(t.quarterly_income_stmt)
        if q is not None and len(q) >= 5:
            qg = _growth(q.iloc[0], q.iloc[4])
    except Exception:
        pass
    if qg is None and info.get("earningsQuarterlyGrowth") is not None:
        qg = info["earningsQuarterlyGrowth"]
    res["c_growth"] = qg

    # A: annual earnings growth
    cagr, all_up = None, False
    try:
        a = _series_from_statement(t.income_stmt)
        if a is not None and len(a) >= 3:
            a = a.iloc[:4]
            years = len(a) - 1
            newest, oldest = a.iloc[0], a.iloc[-1]
            if oldest > 0 and newest > 0:
                cagr = (newest / oldest) ** (1 / years) - 1
            vals = list(a.iloc[::-1])  # oldest -> newest
            all_up = all(vals[i + 1] > vals[i] for i in range(len(vals) - 1))
    except Exception:
        pass
    res["a_cagr"], res["a_all_up"] = cagr, all_up

    inst = info.get("heldPercentInstitutions")
    res["i_inst"] = inst
    roe = info.get("returnOnEquity")

    res["Qtr EPS Growth %"] = None if qg is None else round(qg * 100, 1)
    res["Annual EPS CAGR %"] = None if cagr is None else round(cagr * 100, 1)
    res["Inst. Held %"] = None if inst is None else round(inst * 100, 1)
    res["ROE %"] = None if roe is None else round(roe * 100, 1)
    return res


# ----------------------------------------------------------------------------
# Main screen
# ----------------------------------------------------------------------------
def screen(tickers, top_n, max_fundamentals):
    prices = download_prices(tickers)
    print(f"  Got price history for {len(prices)} of {len(tickers)} stocks.")

    rows = []
    for t, df in prices.items():
        try:
            m = price_metrics(df)
        except Exception:
            m = None
        if m:
            m["Ticker"] = t
            rows.append(m)
    if not rows:
        raise RuntimeError("No usable price data. Check your internet connection or tickers.")

    pm = pd.DataFrame(rows)
    # L: relative strength rating 1-99 versus everything screened
    pm["RS"] = (pm["rs_raw"].rank(pct=True) * 98 + 1).round().astype(int)
    pm["L"] = pm["RS"] >= L_MIN_RS_RATING

    # Fundamentals are slow, so only fetch them for the strongest stocks
    shortlist = pm.sort_values(["L", "N", "RS"], ascending=False).head(max_fundamentals)
    print(f"  Checking earnings and ownership for the top {len(shortlist)} by price strength...")

    results = []
    for i, row in enumerate(shortlist.to_dict("records"), 1):
        t = row["Ticker"]
        if sys.stdout.isatty():
            print(f"    [{i}/{len(shortlist)}] {t}      ", end="\r")
        elif i % 10 == 0:
            print(f"    {i}/{len(shortlist)} checked")
        try:
            f = fundamentals(t)
        except Exception:
            f = {"Name": t, "c_growth": None, "a_cagr": None, "a_all_up": False,
                 "i_inst": None, "Qtr EPS Growth %": None,
                 "Annual EPS CAGR %": None, "Inst. Held %": None, "ROE %": None}
        time.sleep(0.2)  # be gentle with Yahoo

        C = f["c_growth"] is not None and f["c_growth"] >= C_MIN_QTR_EPS_GROWTH
        A = (f["a_cagr"] is not None and f["a_cagr"] >= A_MIN_ANNUAL_EPS_CAGR
             and f["a_all_up"])
        I = f["i_inst"] is not None and f["i_inst"] >= I_MIN_INST_HOLDING

        flags = {"C": C, "A": A, "N": bool(row["N"]), "S": bool(row["S"]),
                 "L": bool(row["L"]), "I": I}
        passed = "".join(k if v else "·" for k, v in flags.items())
        results.append({
            "Ticker": t,
            "Name": f["Name"],
            "Score": sum(flags.values()),
            "Passes": passed,
            "RS": row["RS"],
            "Price": row["Price"],
            "% Below High": row["% Below High"],
            "Qtr EPS Growth %": f["Qtr EPS Growth %"],
            "Annual EPS CAGR %": f["Annual EPS CAGR %"],
            "ROE %": f["ROE %"],
            "Up/Down Vol": row["Up/Down Vol"],
            "Inst. Held %": f["Inst. Held %"],
            "12m Return %": row["12m Return %"],
        })
    if sys.stdout.isatty():
        print(" " * 40, end="\r")

    out = pd.DataFrame(results)
    out = out.sort_values(["Score", "RS"], ascending=False).reset_index(drop=True)
    out.index = out.index + 1
    return out.head(top_n), out


def _clean(v):
    """Make a value safe for JSON (no NaN, no numpy types)."""
    if v is None:
        return None
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def to_json_rows(df):
    rows = []
    for rank, r in df.iterrows():
        rows.append({
            "rank": int(rank),
            "ticker": r["Ticker"],
            "name": r["Name"],
            "score": int(r["Score"]),
            "passes": {k: (k in r["Passes"]) for k in "CANSLI"},
            "rs": _clean(r["RS"]),
            "price": _clean(r["Price"]),
            "pct_below_high": _clean(r["% Below High"]),
            "qtr_eps_growth": _clean(r["Qtr EPS Growth %"]),
            "annual_eps_cagr": _clean(r["Annual EPS CAGR %"]),
            "roe": _clean(r["ROE %"]),
            "updown_vol": _clean(r["Up/Down Vol"]),
            "inst_held": _clean(r["Inst. Held %"]),
            "return_12m": _clean(r["12m Return %"]),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description="CAN SLIM stock screener")
    p.add_argument("--market", choices=["asx", "us", "both"], default="asx")
    p.add_argument("--tickers-file", help="text file of tickers (use .AX for ASX)")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--max-fundamentals", type=int, default=60,
                   help="how many of the strongest stocks to check earnings for, per market")
    p.add_argument("--json", help="also write results to this JSON file (used by the phone app)")
    args = p.parse_args()

    print("\nCAN SLIM screener")
    print("=================")

    # Build a ticker list per market; relative strength is ranked within each market
    universes = {}
    if args.tickers_file:
        tickers = load_tickers_file(args.tickers_file)
        print(f"Loaded {len(tickers)} tickers from {args.tickers_file}")
        asx = [t for t in tickers if t.endswith(".AX")]
        us = [t for t in tickers if not t.endswith(".AX")]
        if asx:
            universes["asx"] = asx
        if us:
            universes["us"] = us
    else:
        markets = ["asx", "us"] if args.market == "both" else [args.market]
        if "asx" in markets:
            print("Getting ASX 200 list...")
            universes["asx"] = get_asx_universe()
        if "us" in markets:
            print("Getting S&P 500 list...")
            universes["us"] = get_us_universe()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "thresholds": {
            "C": f"Quarterly EPS up {C_MIN_QTR_EPS_GROWTH:.0%}+ year on year",
            "A": f"Annual EPS growth {A_MIN_ANNUAL_EPS_CAGR:.0%}+ a year, no down years",
            "N": f"Within {N_MAX_BELOW_HIGH:.0%} of 52-week high, above 50-day average",
            "S": "More volume on up days than down days (50 days), liquid",
            "L": f"Relative strength rating {L_MIN_RS_RATING}+",
            "I": f"Institutions hold {I_MIN_INST_HOLDING:.0%}+ of shares",
            "M": "Index above its 50- and 200-day averages",
        },
        "markets": {},
    }
    saved_frames = []

    for mk, tickers in universes.items():
        sym, name = INDEXES[mk]
        tickers = list(dict.fromkeys(tickers))
        print(f"\n=== {name} ===")
        status, detail = market_direction(sym)
        print(f"M - Market direction: {status}  ({detail})")
        if status == "Downtrend":
            print("  -> CAN SLIM says avoid new buys while the market is in a downtrend.")

        print(f"Screening {len(tickers)} stocks...")
        try:
            top, full = screen(tickers, args.top, args.max_fundamentals)
        except Exception as e:
            print(f"  Screen failed for {name}: {e}")
            continue

        print(f"\nTop {len(top)} by CAN SLIM score (max 6; M is shown above)")
        print("Passes column: letters met, · = not met or no data\n")
        print(top.to_string(na_rep="-"))

        full = full.copy()
        full.insert(0, "Market", name)
        saved_frames.append(full)
        report["markets"][mk] = {
            "name": name,
            "index_status": status,
            "index_detail": detail,
            "screened": len(tickers),
            "top": to_json_rows(top),
        }

    # Keep the last good results for any market that failed this time
    if args.json and os.path.exists(args.json):
        try:
            with open(args.json) as f:
                previous = json.load(f).get("markets", {})
            for mk in universes:
                if mk not in report["markets"] and mk in previous:
                    print(f"Keeping previous results for {INDEXES[mk][1]}.")
                    report["markets"][mk] = previous[mk]
        except Exception:
            pass

    if not saved_frames:
        sys.exit("No markets could be screened.")

    fname = f"canslim_results_{date.today():%Y%m%d}.csv"
    pd.concat(saved_frames).to_csv(fname)
    print(f"\nFull shortlist saved to {fname}")

    if args.json:
        tmp = args.json + ".tmp"
        with open(tmp, "w") as f:
            json.dump(report, f, indent=1)
        os.replace(tmp, args.json)
        print(f"App results saved to {args.json}")

    print("Not financial advice. Free data has gaps; check figures before acting.\n")


if __name__ == "__main__":
    main()
