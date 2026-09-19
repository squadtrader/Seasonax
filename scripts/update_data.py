"""Calcule la saisonnalité des 7 devises majeures et écrit docs/data.json."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# Pour ajouter un actif plus tard : une ligne ici suffit.
PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X",
}

OUT = Path(__file__).resolve().parent.parent / "docs" / "data.json"


def fetch_close(ticker: str) -> pd.Series:
    """Historique journalier maximal des clôtures, jours ouvrés uniquement."""
    df = yf.download(ticker, period="max", interval="1d",
                     auto_adjust=False, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"Aucune donnée pour {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Close"].dropna()
    s = s[s > 0]
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[s.index.dayofweek < 5]
    return s[~s.index.duplicated()].sort_index()


def stats(values) -> dict:
    v = pd.Series(list(values), dtype="float64").dropna()
    if v.empty:
        return {"n": 0, "mean": None, "median": None,
                "pct_up": None, "best": None, "worst": None}
    return {
        "n": int(len(v)),
        "mean": round(float(v.mean()), 4),
        "median": round(float(v.median()), 4),
        "pct_up": round(float((v > 0).mean() * 100), 1),
        "best": round(float(v.max()), 3),
        "worst": round(float(v.min()), 3),
    }


def analyze(close: pd.Series, now: pd.Timestamp) -> dict:
    today = now.normalize()

    # Une clôture datée d'aujourd'hui avant 22h UTC n'est pas définitive.
    if close.index[-1] == today and now.hour < 22:
        close = close.iloc[:-1]

    first_date, last_date = close.index[0], close.index[-1]

    # --- Par année : variation de clôture à clôture ---------------------
    g = close.groupby(close.index.year)
    last, first = g.last(), g.first()
    ref = last.shift(1).fillna(first)
    year_ret = (last / ref - 1) * 100
    first_partial = first_date > pd.Timestamp(first_date.year, 1, 10)
    by_year = []
    for y, r in year_ret.items():
        partial = (y == first_date.year and first_partial) or y == today.year
        by_year.append({"year": int(y), "change": round(float(r), 3),
                        "partial": bool(partial)})

    # --- Par mois de l'année (mois terminés uniquement) -----------------
    ym = close.groupby([close.index.year, close.index.month]).last()
    mret = (ym.pct_change() * 100).dropna()
    mret = mret[[k != (today.year, today.month) for k in mret.index]]
    months = mret.index.get_level_values(1)
    by_month = [{"month": m, **stats(mret[months == m])} for m in range(1, 13)]

    matrix = []
    for y in sorted(set(mret.index.get_level_values(0))):
        row = []
        for m in range(1, 13):
            v = mret.get((y, m), np.nan)
            row.append(None if pd.isna(v) else round(float(v), 3))
        matrix.append({"year": int(y), "values": row})

    # --- Par semaine du mois (blocs de jours : 1-7, 8-14, 15-21, 22-28, 29+)
    lr = np.log(close).diff().dropna()
    d = pd.DataFrame({"lr": lr.values, "y": lr.index.year,
                      "m": lr.index.month, "w": (lr.index.day - 1) // 7 + 1},
                     index=lr.index)
    d = d[~((d.y == d.y.iloc[0]) & (d.m == d.m.iloc[0]))]      # 1er mois partiel
    d = d[~((d.y == today.year) & (d.m == today.month))]       # mois en cours
    wk = np.expm1(d.groupby(["y", "m", "w"])["lr"].sum()) * 100
    weeks = wk.index.get_level_values("w")
    by_week = [{"week": w, **stats(wk[weeks == w])} for w in range(1, 6)]

    # --- Par jour de la semaine -----------------------------------------
    dr = (close.pct_change() * 100).dropna()
    by_weekday = [{"weekday": k, **stats(dr[dr.index.dayofweek == k])}
                  for k in range(5)]

    return {
        "start": first_date.date().isoformat(),
        "end": last_date.date().isoformat(),
        "years": round((last_date - first_date).days / 365.25, 1),
        "sessions": int(len(close)),
        "last_close": round(float(close.iloc[-1]), 5),
        "by_year": by_year,
        "by_month": by_month,
        "month_matrix": matrix,
        "by_week": by_week,
        "by_weekday": by_weekday,
    }


def main() -> None:
    previous = {}
    if OUT.exists():
        try:
            previous = json.loads(OUT.read_text(encoding="utf-8")).get("pairs", {})
        except Exception:
            pass

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    pairs, failures = {}, []
    for name, ticker in PAIRS.items():
        try:
            pairs[name] = {"ticker": ticker, **analyze(fetch_close(ticker), now)}
            print(f"OK   {name}: {pairs[name]['start']} -> {pairs[name]['end']}")
        except Exception as e:                       # on garde l'ancienne version
            failures.append(name)
            print(f"FAIL {name}: {e}", file=sys.stderr)
            if name in previous:
                pairs[name] = previous[name]

    if not pairs or len(failures) == len(PAIRS):
        sys.exit("Aucune paire n'a pu être mise à jour.")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": "Yahoo Finance", "pairs": pairs}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")
    print(f"Écrit : {OUT}")
    if failures:
        print("Paires non mises à jour :", ", ".join(failures), file=sys.stderr)


if __name__ == "__main__":
    main()
