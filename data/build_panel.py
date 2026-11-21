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
V = "2026-11-20"                       # download date
COLUMNS = ["country", "year", "delta_E", "gdp_growth", "debt_gdp", "debt_gdp_lag", "prim_bal_lag", "ca_gdp",
           "infl_diff", "wui", "vix", "shadow_rate", "rate_spread", "pi_stress_annual",
           "delta_debt_pos_lag", "delta_debt_neg_lag", "prim_bal_streak_lag", "gepu"]


def raw(name):
    return os.path.join(RAW, name)


# economies
def economy(iso, ifs, weo, bis, cpi=None, wui=None, rate=None, weo_early=()):
    return dict(iso=iso, cpi=cpi or [("ifs", ifs, 12)], weo=weo, wui=wui or [weo], weo_early=weo_early,
                rate=rate or [("bis", bis)])


def imf_cpi(iso, freq="M", transformation="YOY_PCH_PA_PT"):
    return ("imf", iso, transformation, freq)


EURO_WUI = ["AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "NLD", "PRT", "ESP", "GRC", "SVN", "SVK", "LTU", "LVA"]

ECONOMIES = {
    "Australia": economy("AUS", "Australia", "AUS", "AU: Australia", cpi=[imf_cpi("AUS", "Q")]),
    "Brazil": economy("BRA", "Brazil", "BRA", "BR: Brazil"),
    "Canada": economy("CAN", "Canada", "CAN", "CA: Canada"),
    "Czechia": economy("CZE", "Czech Republic", "CZE", "CZ: Czechia", cpi=[imf_cpi("CZE")]),
    "Euro Area": economy("G163", "Euro Area (EA)", "EURO", "XM: Euro area", cpi=[("hicp",)], wui=EURO_WUI),
    "Hungary": economy("HUN", "Hungary", "HUN", "HU: Hungary", cpi=[imf_cpi("HUN")]),
    "India": economy("IND", "India", "IND", "IN: India", cpi=[("ifs", "India", 12), imf_cpi("IND")]),
    "Indonesia": economy("IDN", "Indonesia", "IDN", "ID: Indonesia", cpi=[("ifs", "Indonesia", 12), imf_cpi("IDN")],
        rate=[("bis", "ID: Indonesia"), ("fred", "IRSTCI01IDM156N")]),
    "Israel": economy("ISR", "Israel", "ISR", "IL: Israel", cpi=[imf_cpi("ISR")]),
    "Japan": economy("JPN", "Japan", "JPN", "JP: Japan", cpi=[("ifs", "Japan", 12), imf_cpi("JPN")],
        rate=[("bis", "JP: Japan"), ("fred", "IRSTCI01JPM156N")]),
    "Korea": economy("KOR", "Korea, Republic of", "KOR", "KR: Korea", weo_early=("GGXWDG_NGDP", "GGXONLB_G01_GDP_PT")),
    "Mexico": economy("MEX", "Mexico", "MEX", "MX: Mexico"),
    "New Zealand": economy("NZL", "New Zealand", "NZL", "NZ: New Zealand", cpi=[imf_cpi("NZL", "Q")]),
    "Norway": economy("NOR", "Norway", "NOR", "NO: Norway"),
    "Poland": economy("POL", "Poland, Republic of", "POL", "PL: Poland"),
    "Russia": economy("RUS", "Russian Federation", "RUS", "RU: Russia", cpi=[imf_cpi("RUS")]),
    "South Africa": economy("ZAF", "South Africa", "ZAF", "ZA: South Africa"),
    "Sweden": economy("SWE", "Sweden", "SWE", "SE: Sweden"),
    "Switzerland": economy("CHE", "Switzerland", "CHE", "CH: Switzerland"),
    "Thailand": economy("THA", "Thailand", "THA", "TH: Thailand", cpi=[imf_cpi("THA", "M", "SRP_YOY_PCH_PA_PT")]),
    "UK": economy("GBR", "United Kingdom", "GBR", "GB: United Kingdom"),
}
US_CPI = ("ifs", "United States", 12)
US_RATE = [("bis", "US: United States")]           # Fed target-range midpoint
ZLB_YEARS = list(range(2009, 2016)) + [2020, 2021]


# readers
@cache
def ifs_cpi_rows():
    d = pd.read_csv(raw("ifs_cpi.csv"))
    d = d[d.TYPE_OF_TRANSFORMATION.str.contains("Year-over-year", na=False)]
    return d.drop_duplicates("COUNTRY").set_index("COUNTRY")


# series of an IMF SDMX download
@cache
def sdmx(file):
    text = open(raw(file), encoding="utf-8").read()
    return [(dict(re.findall(r'(\w+)="([^"]*)"', attrs)), dict(re.findall(r'TIME_PERIOD="([^"]+)"\s+OBS_VALUE="([^"]+)"', body)))
            for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", text, flags=re.S)]


# {year: value} from an IMF download
def imf(file, period, **keys):
    [obs] = [o for a, o in sdmx(file) if all(a.get(k) == v for k, v in keys.items())]
    return values((p[:4], v) for p, v in obs.items() if p.endswith("-" + period))


@cache
def bis_rows():
    return pd.read_csv(raw("bis_cbpol_monthly.csv"), usecols=[4, 5, 6], names=["area", "period", "value"], header=0)


# a WUI workbook sheet
@cache
def wui_sheet(name):
    rows = list(openpyxl.load_workbook(raw("wui.xlsx"), read_only=True)[name].iter_rows(values_only=True))
    return pd.DataFrame([r[1:] for r in rows[1:] if r[0]], index=[r[0] for r in rows[1:] if r[0]],
                        columns=rows[0][1:]).apply(pd.to_numeric, errors="coerce")


@cache
def weo_vintage(vintage, ind):
    return json.load(open(raw(f"weo_{vintage}_{ind}.json")))["values"][ind]


def values(pairs):
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


# each year takes the first series with a value
def first_available(series):
    out = {}
    for s in series:
        for y, v in s.items():
            out.setdefault(y, v)
    return out


# economy series
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


# year mean of quarters, GDP-weighted across codes
def wui(codes):
    x, w = wui_sheet("T2")[codes], wui_sheet("T9")
    q = {}
    for label, row in x.dropna(how="all").iterrows():
        if "q" not in str(label) or int(str(label)[:4]) not in YEARS:
            continue
        year, row = int(str(label)[:4]), row.dropna()
        value = row.iloc[0] if len(codes) == 1 else (row * w.loc[year, row.index]).sum() / w.loc[year, row.index].sum()
        q.setdefault(year, []).append(float(value))
    return {y: float(np.mean(v)) for y, v in q.items()}


# WEO block
WEO_SERIES = ["GGXWDG_NGDP", "NGDP_RPCH", "BCA_NGDPD", "GGXONLB_G01_GDP_PT"]


# October 2026 series, April 2026 before October's start
def weo_series(ind, code, early=()):
    s = {int(y): float(v) for y, v in weo_vintage("oct2026", ind).get(code, {}).items()}
    if ind in early:
        apr = {int(y): float(v) for y, v in weo_vintage("apr2026", ind)[code].items()}
        start = min(s)
        if abs(s[start] - apr[start]) > 0.25:
            raise SystemExit(f"{code} {ind}: April and October 2026 differ by more than 0.25 at {start}")
        s.update({y: v for y, v in apr.items() if 1999 <= y < start})
    return s


# WEO-derived columns, 2001-2025
def weo_columns(name):
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


# global series: annual mean stress probability
def stress_probability(monthly_vix):
    # start values: calm and stressed VIX
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


# panel
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
