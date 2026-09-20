#!/usr/bin/env python3
"""
Calcule la saisonnalité des 7 paires de devises majeures et écrit docs/data.json.

Découpages calculés pour chaque paire :
  - par année
  - par mois de l'année
  - par semaine du mois   (S1 = jours 1-7, S2 = 8-14, S3 = 15-21, S4 = 22-28, S5 = 29-31)
  - par jour de la semaine (lundi à vendredi)

Toutes les variations sont en % et calculées à partir des cours de clôture
journaliers. Une période "composée" = variation cumulée des jours qu'elle contient.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "data.json"

# Pour ajouter un actif plus tard (indice, or, pétrole...), ajouter une ligne ici.
PAIRS = {
    "EURUSD": {"label": "EUR/USD", "ticker": "EURUSD=X"},
    "GBPUSD": {"label": "GBP/USD", "ticker": "GBPUSD=X"},
    "USDJPY": {"label": "USD/JPY", "ticker": "USDJPY=X"},
    "USDCHF": {"label": "USD/CHF", "ticker": "USDCHF=X"},
    "USDCAD": {"label": "USD/CAD", "ticker": "USDCAD=X"},
    "AUDUSD": {"label": "AUD/USD", "ticker": "AUDUSD=X"},
    "NZDUSD": {"label": "NZD/USD", "ticker": "NZDUSD=X"},
}

MONTHS = ["Janvier", "Février", "Mars", "Avril", "Mai", "Juin", "Juillet",
          "Août", "Septembre", "Octobre", "Novembre", "Décembre"]
WEEKDAYS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi"]
WEEK_RANGES = ["jours 1 à 7", "jours 8 à 14", "jours 15 à 21", "jours 22 à 28", "jours 29 à 31"]


# --------------------------------------------------------------------------- #
# Téléchargement
# --------------------------------------------------------------------------- #
def fetch_close(ticker: str, retries: int = 3):
    """Cours de clôture journaliers (jours terminés uniquement) + liste des erreurs retirées."""
    import yfinance as yf

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            df = yf.Ticker(ticker).history(period="max", interval="1d", auto_adjust=False)
            if df is None or df.empty:
                raise RuntimeError("aucune donnée reçue")
            close = df["Close"].copy()
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            return remove_spikes(clean_close(close))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"  tentative {attempt}/{retries} échouée pour {ticker} : {exc}")
            time.sleep(3 * attempt)
    raise RuntimeError(f"{ticker} : {last_error}")


def clean_close(close: pd.Series) -> pd.Series:
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close = close[close.notna() & (close > 0)]
    close = close[close.index.dayofweek < 5]
    # On retire la bougie du jour en cours (incomplète) : seulement des jours terminés.
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    return close[close.index < today]


SPIKE_THRESHOLD = 0.05   # variation d'un jour à partir de laquelle on soupçonne une erreur
SPIKE_REVERSAL = 0.25    # ... si le lendemain annule presque tout (écart net < 25 % du saut)


def remove_spikes(close: pd.Series):
    """
    Yahoo contient parfois un cours isolé aberrant (ex. +17 % un jour, puis retour à la normale
    le lendemain). On retire ces points : un vrai mouvement (Brexit, franc suisse en 2015...)
    ne s'annule pas le jour suivant, donc il est conservé.
    """
    r = close.pct_change()
    nxt = r.shift(-1)
    net = (1 + r) * (1 + nxt) - 1
    spike = (r.abs() > SPIKE_THRESHOLD) & (np.sign(r) != np.sign(nxt)) & (net.abs() < SPIKE_REVERSAL * r.abs())
    removed = [{"date": d.strftime("%Y-%m-%d"), "change": round(float(r[d] * 100), 2)} for d in close.index[spike.fillna(False)]]
    return close[~spike.fillna(False)], removed


# --------------------------------------------------------------------------- #
# Statistiques
# --------------------------------------------------------------------------- #
def r(x, digits=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), digits)


def compound(x: pd.Series) -> float:
    """Variation cumulée (%) d'une série de rendements journaliers (%)."""
    return (np.prod(1 + x.to_numpy() / 100) - 1) * 100


def stat_block(values: pd.Series) -> dict:
    v = values.dropna()
    n = len(v)
    if n == 0:
        return {"mean": None, "median": None, "pct_up": None, "best": None, "worst": None, "n": 0}
    return {
        "mean": r(v.mean()),
        "median": r(v.median()),
        "pct_up": r(100 * (v > 0).mean(), 1),
        "best": r(v.max()),
        "worst": r(v.min()),
        "n": int(n),
    }


def bucket_key(ts: pd.Timestamp, kind: str):
    if kind == "month":
        return (ts.year, ts.month)
    return (ts.year, ts.month, (ts.day - 1) // 7 + 1)


def drop_incomplete(groups: pd.Series, close: pd.Series, kind: str) -> pd.Series:
    """Retire la première période (début de l'historique) et la dernière si elle est en cours."""
    keys_to_drop = {bucket_key(close.index[0], kind)}
    last = close.index[-1]
    next_day = last + pd.offsets.BDay(1)
    if bucket_key(next_day, kind) == bucket_key(last, kind):
        keys_to_drop.add(bucket_key(last, kind))
    return groups[[k not in keys_to_drop for k in groups.index]]


def week_key(ts: pd.Timestamp):
    """(année, N) où la semaine est celle dont le lundi est le Nème lundi de l'année."""
    monday = ts - pd.Timedelta(days=int(ts.dayofweek))
    return (monday.year, (monday.dayofyear - 1) // 7 + 1)


def drop_incomplete_weeks(weeks: pd.Series, close: pd.Series) -> pd.Series:
    drop = {week_key(close.index[0])}
    last = close.index[-1]
    if week_key(last + pd.offsets.BDay(1)) == week_key(last):
        drop.add(week_key(last))
    return weeks[[k not in drop for k in weeks.index]]


def compute_pair(close: pd.Series) -> dict:
    ret = close.pct_change().dropna() * 100
    df = pd.DataFrame({"ret": ret})
    df["year"] = df.index.year
    df["month"] = df.index.month
    df["wom"] = (df.index.day - 1) // 7 + 1
    df["wd"] = df.index.dayofweek

    first_year, last_year = close.index[0].year, close.index[-1].year

    # ---- Par année -------------------------------------------------------- #
    yearly = []
    for year, g in df.groupby("year")["ret"]:
        partial = (year == first_year and close.index[0] > pd.Timestamp(year, 1, 10)) or (
            year == last_year and close.index[-1] < pd.Timestamp(year, 12, 24))
        yearly.append({"year": int(year), "ret": r(compound(g)), "partial": bool(partial)})
    full_years = pd.Series([y["ret"] for y in yearly if not y["partial"]], dtype=float)

    # ---- Par mois de l'année --------------------------------------------- #
    monthly_series = df.groupby(["year", "month"])["ret"].apply(compound)
    monthly_series = drop_incomplete(monthly_series, close, "month")

    monthly = []
    for m in range(1, 13):
        vals = monthly_series[monthly_series.index.get_level_values("month") == m]
        monthly.append({"key": m, "label": MONTHS[m - 1], **stat_block(vals)})

    matrix = {}
    for (year, month), value in monthly_series.items():
        matrix.setdefault(str(year), [None] * 12)[month - 1] = r(value, 2)

    # ---- Par semaine du mois --------------------------------------------- #
    wom_series = df.groupby(["year", "month", "wom"])["ret"].apply(compound)
    wom_series = drop_incomplete(wom_series, close, "wom")

    week_of_month = []
    for w in range(1, 6):
        vals = wom_series[wom_series.index.get_level_values("wom") == w]
        week_of_month.append({"key": w, "label": f"Semaine {w}", "range": WEEK_RANGES[w - 1],
                              **stat_block(vals)})

    # ---- Par jour de la semaine ------------------------------------------ #
    weekday = []
    for d in range(5):
        weekday.append({"key": d + 1, "label": WEEKDAYS[d], **stat_block(df.loc[df["wd"] == d, "ret"])})

    # ---- Par semaine de l'année (Nème lundi) et Nème jour de la semaine -------- #
    # Le Nème lundi de l'année tombe toujours entre le jour 7N-6 et 7N de l'année.
    # Idem pour le Nème mardi, mercredi, etc. : N = (jour de l'année - 1) // 7 + 1.
    df["nth"] = (df.index.dayofyear - 1) // 7 + 1
    monday = df.index - pd.to_timedelta(df.index.dayofweek, unit="D")
    df["wk_year"] = np.asarray(monday.year)
    df["wk_n"] = np.asarray((monday.dayofyear - 1) // 7 + 1)
    wk_series = drop_incomplete_weeks(df.groupby(["wk_year", "wk_n"])["ret"].apply(compound), close)

    week_of_year = []
    for n in range(1, 54):
        sub = df[df["nth"] == n]
        if sub.empty:
            continue
        wvals = wk_series[wk_series.index.get_level_values("wk_n") == n]
        md = [(d.month, d.day) for d in sub.index]
        lo, hi = min(md), max(md)
        days = [{"weekday": d + 1, **stat_block(sub.loc[sub["wd"] == d, "ret"])} for d in range(5)]
        week_of_year.append({"key": n, "from": f"{lo[0]:02d}-{lo[1]:02d}", "to": f"{hi[0]:02d}-{hi[1]:02d}",
                             **stat_block(wvals), "days": days})

    # ---- Calendrier : combien de lundis, mardis... par année -------------------- #
    partial_by_year = {y["year"]: y["partial"] for y in yearly}
    year_calendar = []
    for y in range(first_year, last_year + 1):
        bdays = pd.bdate_range(f"{y}-01-01", f"{y}-12-31")
        year_calendar.append({
            "year": y,
            "counts": [int((bdays.dayofweek == d).sum()) for d in range(5)],
            "trading_days": int((close.index.year == y).sum()),
            "partial": bool(partial_by_year.get(y, False)),
        })

    start, end = close.index[0], close.index[-1]
    return {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "years": round((end - start).days / 365.25, 1),
        "days": int(len(close)),
        "last_close": r(close.iloc[-1], 5),
        "yearly": yearly,
        "yearly_stats": stat_block(full_years),
        "monthly": monthly,
        "monthly_matrix": matrix,
        "week_of_month": week_of_month,
        "weekday": weekday,
        "week_of_year": week_of_year,
        "year_calendar": year_calendar,
    }


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def load_previous() -> dict:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def main() -> int:
    previous = load_previous()
    previous_pairs = previous.get("pairs", {})
    pairs_out, failures = {}, []

    for code, info in PAIRS.items():
        print(f"{info['label']} ({info['ticker']})")
        try:
            close, removed = fetch_close(info["ticker"])
            stats = compute_pair(close)
            stats["removed_outliers"] = removed
            print(f"  {stats['days']} jours, du {stats['start']} au {stats['end']}")
            if removed:
                print(f"  cours aberrants retirés : {removed}")
            pairs_out[code] = {"label": info["label"], "ticker": info["ticker"], **stats}
        except Exception as exc:  # noqa: BLE001
            print(f"  ÉCHEC : {exc}")
            failures.append(code)
            if code in previous_pairs:  # on garde les anciennes données plutôt que de tout perdre
                pairs_out[code] = previous_pairs[code]

    if not pairs_out:
        print("Aucune donnée disponible, rien n'est écrit.")
        return 1

    payload = {"source": "Yahoo Finance", "pairs": pairs_out}
    old_payload = {k: v for k, v in previous.items() if k != "generated"}
    if old_payload == payload:
        print("Aucun changement dans les données.")
    else:
        payload["generated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                       encoding="utf-8")
        print(f"Écrit : {OUT}")

    if failures:
        print(f"Paires en échec : {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
