import json
import os
import re
from functools import cache

import numpy as np
import openpyxl
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OUT = os.path.join(HERE, "panel", "panel21.csv")
YEARS = range(2001, 2026)
V = "2026-11-20"                       # retrieval date of the November 2026 downloads
COLUMNS = ["country", "year", "delta_E", "gdp_growth", "debt_gdp", "debt_gdp_lag", "prim_bal_lag", "ca_gdp",
           "infl_diff", "wui", "vix", "shadow_rate", "rate_spread", "pi_stress_annual",
           "delta_debt_pos_lag", "delta_debt_neg_lag", "prim_bal_streak_lag", "gepu"]


def raw(name):
    return os.path.join(RAW, name)


# ------------------------------------------------------------------ economies
# iso is the IMF code, which also selects the exchange rate. Sources, in priority order:
#   cpi   ("ifs", name, month)               ifs_cpi.csv, year-on-year rate of that month
#         ("imf", iso, transformation, freq) imf_cpi, December (M) or fourth-quarter (Q) rate
#         ("hicp",)                          Eurostat euro-area HICP, December
#   rate  ("bis", area)                      BIS central bank policy rate, end of December
#         ("fred", series)                   FRED copy of an OECD series, December
# weo is the DataMapper code; weo_early lists the series whose years before the October 2026 start come
# from April 2026. why gives the reason for every rule that departs from the defaults (the 3 February
# 2026 IFS CPI file, the economy's BIS policy rate, its own WUI and October 2026 WEO series).
def economy(iso, ifs, weo, bis, cpi=None, wui=None, rate=None, weo_early=(), why=None):
    return dict(iso=iso, cpi=cpi or [("ifs", ifs, 12)], weo=weo, wui=wui or [weo], weo_early=weo_early,
                rate=rate or [("bis", bis)], why=why or {})


def imf_cpi(iso, freq="M", transformation="YOY_PCH_PA_PT"):
    return ("imf", iso, transformation, freq)


EURO_WUI = ["AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "NLD", "PRT", "ESP", "GRC", "SVN", "SVK", "LTU", "LVA"]
NOT_IN_FEB_FILE = "not in the 3 February 2026 IFS file; the same IMF dataset, 20 November 2026"
ENDS_NOV_2025 = "the 3 February 2026 IFS file ends in November 2025; December 2025 from the same series, 20 November 2026"

ECONOMIES = {
    "Australia": economy("AUS", "Australia", "AUS", "AU: Australia", cpi=[imf_cpi("AUS", "Q")],
        why=dict(cpi="CPI is quarterly (monthly only from April 2025); " + NOT_IN_FEB_FILE)),
    "Brazil": economy("BRA", "Brazil", "BRA", "BR: Brazil"),
    "Canada": economy("CAN", "Canada", "CAN", "CA: Canada"),
    "Czechia": economy("CZE", "Czech Republic", "CZE", "CZ: Czechia", cpi=[imf_cpi("CZE")], why=dict(cpi=NOT_IN_FEB_FILE)),
    "Euro Area": economy("G163", "Euro Area (EA)", "EURO", "XM: Euro area", cpi=[("hicp",)], wui=EURO_WUI,
        why=dict(cpi="the official euro-area inflation rate, HICP for the changing composition",
                 wui="WUI has no euro-area index: the 15 members it covers that adopted the euro by 2015, weighted by "
                     "the authors' GDP weights")),
    "Hungary": economy("HUN", "Hungary", "HUN", "HU: Hungary", cpi=[imf_cpi("HUN")], why=dict(cpi=NOT_IN_FEB_FILE)),
    "India": economy("IND", "India", "IND", "IN: India", cpi=[("ifs", "India", 12), imf_cpi("IND")],
        why=dict(cpi=ENDS_NOV_2025)),
    "Indonesia": economy("IDN", "Indonesia", "IDN", "ID: Indonesia", cpi=[("ifs", "Indonesia", 12), imf_cpi("IDN")],
        rate=[("bis", "ID: Indonesia"), ("fred", "IRSTCI01IDM156N")],
        why=dict(cpi=ENDS_NOV_2025, rate="no BIS policy rate before the BI Rate began in July 2005, so 2001-2004 take "
                                         "the December overnight call rate, as in the earlier work")),
    "Israel": economy("ISR", "Israel", "ISR", "IL: Israel", cpi=[imf_cpi("ISR")], why=dict(cpi=NOT_IN_FEB_FILE)),
    "Japan": economy("JPN", "Japan", "JPN", "JP: Japan", cpi=[("ifs", "Japan", 12), imf_cpi("JPN")],
        rate=[("bis", "JP: Japan"), ("fred", "IRSTCI01JPM156N")],
        why=dict(cpi=ENDS_NOV_2025,
                 rate="no BIS policy rate in 2001-2005 and 2013-2015, when the Bank of Japan targeted reserves or "
                      "the monetary base, so those years take the December uncollateralized overnight call rate (its "
                      "operating target in the other years), as in the earlier work")),
    "Korea": economy("KOR", "Korea, Republic of", "KOR", "KR: Korea", weo_early=("GGXWDG_NGDP", "GGXONLB_G01_GDP_PT"),
        why=dict(weo="the October 2026 WEO starts Korea's debt and primary balance in 2012")),
    "Mexico": economy("MEX", "Mexico", "MEX", "MX: Mexico"),
    "New Zealand": economy("NZL", "New Zealand", "NZL", "NZ: New Zealand", cpi=[imf_cpi("NZL", "Q")],
        why=dict(cpi="CPI is quarterly; " + NOT_IN_FEB_FILE)),
    "Norway": economy("NOR", "Norway", "NOR", "NO: Norway"),
    "Poland": economy("POL", "Poland, Republic of", "POL", "PL: Poland"),
    "Russia": economy("RUS", "Russian Federation", "RUS", "RU: Russia", cpi=[imf_cpi("RUS")], why=dict(cpi=NOT_IN_FEB_FILE)),
    "South Africa": economy("ZAF", "South Africa", "ZAF", "ZA: South Africa"),
    "Sweden": economy("SWE", "Sweden", "SWE", "SE: Sweden"),
    "Switzerland": economy("CHE", "Switzerland", "CHE", "CH: Switzerland"),
    "Thailand": economy("THA", "Thailand", "THA", "TH: Thailand", cpi=[imf_cpi("THA", "M", "SRP_YOY_PCH_PA_PT")],
        why=dict(cpi="the 3 February 2026 file starts Thailand in 2011 and is wrong in 2022-2024; the IMF's 2010=100 "
                     "series, equal to the national-base series wherever both exist")),
    "UK": economy("GBR", "United Kingdom", "GBR", "GB: United Kingdom"),
}
US_CPI = ("ifs", "United States", 12)
US_RATE = [("bis", "US: United States")]           # the midpoint of the Federal Reserve's target range
ZLB_YEARS = list(range(2009, 2016)) + [2020, 2021]


# ------------------------------------------------------------------ readers
@cache
def ifs_cpi_rows():
    d = pd.read_csv(raw("ifs_cpi.csv"))
    d = d[d.TYPE_OF_TRANSFORMATION.str.contains("Year-over-year", na=False)]
    return d.drop_duplicates("COUNTRY").set_index("COUNTRY")


@cache
def sdmx(file):
    """Series of an IMF SDMX download, as (attributes, {period: value}) pairs."""
    text = open(raw(file), encoding="utf-8").read()
    return [(dict(re.findall(r'(\w+)="([^"]*)"', attrs)), dict(re.findall(r'TIME_PERIOD="([^"]+)"\s+OBS_VALUE="([^"]+)"', body)))
            for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", text, flags=re.S)]


def imf(file, period, **keys):
    """{year: value} at period M12 or Q4 of the one series in an IMF download that matches keys."""
    [obs] = [o for a, o in sdmx(file) if all(a.get(k) == v for k, v in keys.items())]
    return values((p[:4], v) for p, v in obs.items() if p.endswith("-" + period))


@cache
def bis_rows():
    return pd.read_csv(raw("bis_cbpol_monthly.csv"), usecols=[4, 5, 6], names=["area", "period", "value"], header=0)


@cache
def wui_sheet(name):
    """A sheet of the WUI workbook as a table indexed by its first column (quarter in T2, year in T9)."""
    rows = list(openpyxl.load_workbook(raw("wui.xlsx"), read_only=True)[name].iter_rows(values_only=True))
    return pd.DataFrame([r[1:] for r in rows[1:] if r[0]], index=[r[0] for r in rows[1:] if r[0]],
                        columns=rows[0][1:]).apply(pd.to_numeric, errors="coerce")


@cache
def weo_vintage(vintage, ind):
    return json.load(open(raw(f"weo_{vintage}_{ind}.json")))["values"][ind]


def values(pairs):
    """{year: float}, missing values left out."""
    out = {}
    for y, v in pairs:
        v = pd.to_numeric(v, errors="coerce")
        if pd.notna(v):
            out[int(y)] = float(v)
    return out


def fred_december(series):
    f = pd.read_csv(raw(f"fred_{series}_{V}.csv"), parse_dates=["observation_date"])
    f = f[f.observation_date.dt.month == 12]
    return values(zip(f.observation_date.dt.year, f[series]))


def first_available(series):
    """Each year takes the first series with a value."""
    out = {}
    for s in series:
        for y, v in s.items():
            out.setdefault(y, v)
    return out


# ------------------------------------------------------------------ economy series
def exchange_rate(iso):
    return imf(f"imf_er_eop_{V}.xml", "M12", COUNTRY=iso, INDICATOR="XDC_USD", TYPE_OF_TRANSFORMATION="EOP_RT")


def euro_hicp():
    j = json.load(open(raw(f"eurostat_hicp_ea_{V}.json")))
    period = {i: p for p, i in j["dimension"]["time"]["category"]["index"].items()}
    return values((period[int(i)][:4], v) for i, v in j["value"].items() if period[int(i)].endswith("-12"))


def cpi_source(src):
    if src[0] == "ifs":
        row = ifs_cpi_rows().loc[src[1]]
        return values((y, row.get(f"{y}-M{src[2]:02d}")) for y in YEARS)
    if src[0] == "imf":
        _, iso, transformation, freq = src
        return imf(f"imf_cpi_{V}.xml", "M12" if freq == "M" else "Q4", COUNTRY=iso,
                   TYPE_OF_TRANSFORMATION=transformation, FREQUENCY=freq)
    return euro_hicp()


def rate_source(src):
    if src[0] == "fred":
        return fred_december(src[1])
    b = bis_rows()
    b = b[(b.area == src[1]) & (b.period.str[5:7] == "12")]
    return values(zip(b.period.str[:4], b.value))


def wui(codes):
    """Mean of the year's quarters. With several codes, each quarter is first their mean weighted by the
    authors' GDP weights of that year (sheet T9: shares of world GDP in current US dollars, a centered
    moving average over +/-5 years), the rule behind the authors' own regional indices in sheet T1."""
    x, w = wui_sheet("T2")[codes], wui_sheet("T9")
    q = {}
    for label, row in x.dropna(how="all").iterrows():
        if "q" not in str(label) or int(str(label)[:4]) not in YEARS:
            continue
        year, row = int(str(label)[:4]), row.dropna()
        value = row.iloc[0] if len(codes) == 1 else (row * w.loc[year, row.index]).sum() / w.loc[year, row.index].sum()
        q.setdefault(year, []).append(float(value))
    return {y: float(np.mean(v)) for y, v in q.items()}


# ------------------------------------------------------------------ WEO block
WEO_SERIES = ["GGXWDG_NGDP", "NGDP_RPCH", "BCA_NGDPD", "GGXONLB_G01_GDP_PT"]


def weo_series(ind, code, early=()):
    """October 2026 series. For an indicator in early, the years from 1999 to the October start come
    from April 2026, which must agree with October within 0.25 at the start."""
    s = {int(y): float(v) for y, v in weo_vintage("oct2026", ind).get(code, {}).items()}
    if ind in early:
        apr = {int(y): float(v) for y, v in weo_vintage("apr2026", ind)[code].items()}
        start = min(s)
        if abs(s[start] - apr[start]) > 0.25:
            raise SystemExit(f"{code} {ind}: April and October 2026 differ by more than 0.25 at {start}")
        s.update({y: v for y, v in apr.items() if 1999 <= y < start})
    return s


def weo_columns(name):
    """WEO-derived columns of one economy, 2001-2025, from series that start in 1999 so that every lag
    of 2001 exists."""
    e = ECONOMIES[name]
    yrs = list(range(1999, YEARS[-1] + 1))
    s = {ind: pd.Series(weo_series(ind, e["weo"], e["weo_early"]), dtype=float).reindex(yrs) for ind in WEO_SERIES}
    D, P = s["GGXWDG_NGDP"], s["GGXONLB_G01_GDP_PT"]
    ddl = (D - D.shift(1)).shift(1)
    run, streak = 0, []
    for x in P.diff():                                         # a missing change ends the streak
        run = 0 if pd.isna(x) else (run + 1 if x < 0 else 0)
        streak.append(run)
    cols = pd.DataFrame(dict(gdp_growth=s["NGDP_RPCH"], debt_gdp=D, debt_gdp_lag=D.shift(1), prim_bal_lag=P.shift(1),
                             ca_gdp=s["BCA_NGDPD"], delta_debt_pos_lag=ddl.clip(lower=0),
                             delta_debt_neg_lag=ddl.clip(upper=0),
                             prim_bal_streak_lag=pd.Series(streak, index=yrs, dtype=float).shift(1)))
    return cols.loc[YEARS[0]:]


# ------------------------------------------------------------------ global series
def stress_probability(monthly_vix):
    """Annual mean of the smoothed probability of the high-mean regime."""
    # the starting values put the first regime near calm VIX levels and the second near stressed levels
    fit = MarkovRegression(monthly_vix.to_numpy(), k_regimes=2, switching_variance=True).fit(
        start_params=np.array([0.95, 0.90, 14.0, 26.0, 5.0, 30.0]), disp=False, maxiter=500)
    stress = int(fit.params[3] > fit.params[2])
    p = pd.Series(fit.smoothed_marginal_probabilities[:, stress], index=monthly_vix.index)
    return p.groupby(level="year").mean()


def shadow_rate():
    ff = pd.read_csv(raw(f"fred_FEDFUNDS_{V}.csv"), parse_dates=["observation_date"])
    rate = ff.groupby(ff.observation_date.dt.year).FEDFUNDS.mean().loc[list(YEARS)]
    wx = pd.read_excel(raw("wu_xia_shadow_rate.xlsx"), sheet_name="Data").iloc[:, [0, 2]].set_axis(["date", "wu_xia"], axis=1)
    wx = wx.groupby(pd.to_datetime(wx.date).dt.year).wu_xia
    if not (wx.count().reindex(ZLB_YEARS) == 12).all():
        raise SystemExit("the Wu-Xia file lacks a month of a zero-lower-bound year")
    rate.loc[ZLB_YEARS] = wx.mean().loc[ZLB_YEARS]
    return rate


def global_series():
    v = pd.read_csv(raw("cboe_vix_history.csv"))
    d = pd.to_datetime(v.DATE, format="%m/%d/%Y")
    v = v.assign(year=d.dt.year, month=d.dt.month)[d.dt.year.isin(YEARS)]
    g = pd.read_excel(raw(f"gepu_{V}.xlsx"))
    g = g[pd.to_numeric(g.Year, errors="coerce").isin(YEARS) & g.Month.notna()]
    if not (g.groupby(g.Year.astype(int)).size() == 12).all():
        raise SystemExit("the GEPU file lacks a month of 2001-2025")
    return pd.DataFrame(dict(vix=v.groupby("year").CLOSE.mean(), shadow_rate=shadow_rate(),
                             pi_stress_annual=stress_probability(v.groupby(["year", "month"]).CLOSE.mean()),
                             gepu=g.groupby(g.Year.astype(int)).GEPU_current.mean()))


# ------------------------------------------------------------------ panel
def economy_rows(name, glob, us_cpi, us_rate):
    e = ECONOMIES[name]
    lev = exchange_rate(e["iso"])
    cpi = first_available([cpi_source(s) for s in e["cpi"]])
    rate = first_available([rate_source(s) for s in e["rate"]])
    w = wui(e["wui"])
    f = pd.DataFrame(index=pd.Index(YEARS, name="year"))
    f["delta_E"] = [100 * np.log(lev[y] / lev[y - 1]) if y in lev and y - 1 in lev else np.nan for y in YEARS]
    f["infl_diff"] = [cpi[y] - us_cpi[y] if y in cpi and y in us_cpi else np.nan for y in YEARS]
    f["rate_spread"] = [rate[y] - us_rate[y] if y in rate and y in us_rate else np.nan for y in YEARS]
    f["wui"] = [w.get(y, np.nan) for y in YEARS]
    return f.join(glob).join(weo_columns(name)).reset_index().assign(country=name)


def build():
    glob = global_series()
    us_cpi = cpi_source(US_CPI)
    us_rate = first_available([rate_source(s) for s in US_RATE])
    return pd.concat([economy_rows(n, glob, us_cpi, us_rate) for n in ECONOMIES], ignore_index=True)[COLUMNS]


if __name__ == "__main__":
    panel = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    panel.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(panel)} rows, {panel.country.nunique()} economies, "
          f"{len(panel.dropna())} economy-years with every column")
