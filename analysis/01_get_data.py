"""Step 1: get the raw data from the Eurostat database. Paths are relative to this file.

Every series comes from the Eurostat dissemination API (JSON-stat format) and is
saved as a tidy CSV in data/raw/, one row per country, year and remaining
dimension, with Eurostat's status flag ("b" = break in series, "p" = provisional).

The repository ships the snapshot we used (downloaded 7 November 2026), so the
analysis can be reproduced without internet access. By default this script only
describes the snapshot; `python 01_get_data.py --download` replaces it with
fresh data. Eurostat revises its tables, so fresh data can move the estimates
slightly.
"""
import itertools
import json
import sys
import time
from datetime import date

import pandas as pd
import requests

from tools import COUNTRIES, PATH_RAW

REFRESH = "--download" in sys.argv

# %% 1(a) Which series do we need?
# Each entry is file name -> (Eurostat table, filters). The filters fix every
# dimension except country and time, and at most one or two dimensions that we
# keep as columns (lists below). Definitions are in the README.
API = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/"
DIVISIONS = [f"CP{k:02d}" for k in range(1, 13)]           # the 12 COICOP divisions
HOUSEHOLDS = ["A1", "A1_DCH", "A_GE2_NDCH", "A_GE2_DCH"]

SERIES = {
    # Outcome: in-work at-risk-of-poverty rate, aged 18-64, employees (SAL) and
    # all employed people (EMP), by working time and by household type.
    "inwork_poverty_employees": ("ilc_iw01", {"wstatus": "SAL", "sex": "T", "age": "Y18-64", "unit": "PC"}),
    "inwork_poverty": ("ilc_iw01", {"wstatus": "EMP", "sex": "T", "age": "Y18-64", "unit": "PC"}),
    "inwork_poverty_worktime": ("ilc_iw07", {"worktime": ["FT", "PT"], "unit": "PC"}),
    "inwork_poverty_household": ("ilc_iw02", {"hhcomp": HOUSEHOLDS, "unit": "PC"}),
    # Poverty rates at 40/50/60/70% of the median, by most frequent activity
    # status (employees, retired, unemployed), aged 18 and over.
    "poverty_by_activity": ("ilc_li04", {"wstatus": ["SAL", "RET", "UNE"],
                                         "rskpovth": ["B_40", "B_50", "B_60", "B_70"],
                                         "statinfo": "MED_EI", "sex": "T", "age": "Y_GE18", "unit": "PC"}),
    # Minimum wage: gross monthly statutory minimum wage, January (S1) and July (S2).
    "minwage_nac": ("earn_mw_cur", {"currency": "NAC"}),
    "minwage_pps": ("earn_mw_cur", {"currency": "PPS"}),
    # Kaitz index: minimum wage as % of mean or median gross monthly earnings.
    "kaitz_mean": ("earn_mw_avgr2", {"nace_r2": "B-S", "indic_se": "MMW_MEAN_ME_PP", "unit": "PC"}),
    "kaitz_median": ("earn_mw_avgr2", {"nace_r2": "B-S", "indic_se": "MMW_MED_ME_PP", "unit": "PC"}),
    # Prices: HICP, annual average index (2015 = 100), all items and by division.
    "hicp_index": ("prc_hicp_aind", {"unit": "INX_A_AVG", "coicop": "CP00"}),
    "hicp_divisions": ("prc_hicp_aind", {"unit": "INX_A_AVG", "coicop": DIVISIONS}),
    # Budget shares (per mille) of the poorest and richest income quintiles.
    "hbs_budget_shares": ("hbs_str_t223", {"quant_inc": ["QU1", "QU5"], "coicop": DIVISIONS, "unit": "PM"}),
    # Controls
    "unemployment": ("une_rt_a", {"age": "Y15-74", "sex": "T", "unit": "PC_ACT"}),
    "gdp_pc_real": ("nama_10_pc", {"na_item": "B1GQ", "unit": "CLV20_EUR_HAB"}),
    "parttime_share": ("lfsa_eppga", {"sex": "T", "age": "Y15-64", "unit": "PC"}),
    # Weights and the employment check
    "employees_thousands": ("lfsa_eegaed", {"sex": "T", "age": "Y20-64", "isced11": "TOTAL", "unit": "THS_PER"}),
    "employment_rate_low_educ": ("lfsa_ergaed", {"sex": "T", "age": "Y20-64", "isced11": "ED0-2", "unit": "PC"}),
    # Mechanisms
    "poverty_threshold": ("ilc_li01", {"statinfo": "MED_EI", "hhcomp": "A1", "rskpovth": "B_60", "unit": "NAC"}),
    "poverty_anchored_2008": ("ilc_li22b", {"statinfo": "MED_EI", "rskpovth": "B_60", "sex": "T",
                                            "age": "Y18-64", "unit": "PC"}),
    "deprivation_q1": ("ilc_mdsd02", {"hhcomp": "TOTAL", "quant_inc": "QU1", "unit": "PC"}),
}
print(len(SERIES), "series from", len({v[0] for v in SERIES.values()}), "Eurostat tables")


# %% 1(b) Download one table
# JSON-stat stores all cells in one flat list, in row-major order over the
# dimensions; we rebuild the labels of each cell from that order.
def fetch(code, filters):
    params = {"format": "JSON", "lang": "EN", "geo": COUNTRIES, **filters}
    for attempt in range(6):                     # the API sometimes drops a connection
        try:
            r = requests.get(API + code, params=params, timeout=180)
            r.raise_for_status()
            break
        except requests.RequestException:
            if attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))
    js = r.json()
    dims, sizes = js["id"], js["size"]
    labels = [sorted(js["dimension"][d]["category"]["index"].items(), key=lambda kv: kv[1]) for d in dims]
    keep = [d for d, s in zip(dims, sizes) if s > 1 or d in ("geo", "time")]
    values, status = js.get("value", {}), js.get("status", {})
    rows = []
    for flat, combo in enumerate(itertools.product(*[range(s) for s in sizes])):
        key = str(flat)
        if key not in values and key not in status:
            continue
        cell = {d: labels[k][combo[k]][0] for k, d in enumerate(dims)}
        rows.append({**{d: cell[d] for d in keep}, "value": values.get(key), "flag": status.get(key, "")})
    return pd.DataFrame(rows), js.get("updated", "")


# %% 1(c) Download everything (only with --download) and describe the snapshot
if REFRESH:
    meta = {}
    for name, (code, filters) in SERIES.items():
        df, updated = fetch(code, filters)
        df.to_csv(PATH_RAW / f"{name}.csv", index=False)
        meta[name] = {"eurostat_code": code, "filters": filters, "eurostat_updated": updated,
                      "downloaded": date.today().isoformat(), "rows": len(df)}
        print("downloaded", name)
    (PATH_RAW / "_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

meta = json.loads((PATH_RAW / "_metadata.json").read_text(encoding="utf-8"))
summary = []
for name, m in meta.items():
    df = pd.read_csv(PATH_RAW / f"{name}.csv", dtype={"time": str})
    summary.append({"file": name, "table": m["eurostat_code"], "rows": len(df),
                    "first": df.time.min(), "last": df.time.max(), "countries": df.geo.nunique(),
                    "eurostat_updated": m["eurostat_updated"][:10], "downloaded": m["downloaded"]})
print(pd.DataFrame(summary).to_string(index=False))
# Interpretation:
# All 21 countries are covered by every table. EU-SILC tables run up to the
# 2025 wave, which describes income in 2024. The Kaitz index on the median wage
# exists for fewer countries and years than the one on the mean wage.
