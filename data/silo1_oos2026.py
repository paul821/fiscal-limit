import json, os, subprocess, sys, datetime as dt
import numpy as np, pandas as pd
import silo1_core as core

CACHE = os.path.join(core.HERE, "raw", "oos2026")
VINTAGE = os.environ.get("SILO1_VINTAGE", "2026-11-08")   # download date
CUTOFF = "2026-10-31"                 # cutoff

# currency, BIS area, IMF and FRED codes
CODES = {"Australia": ("AUD", "AU", "AUS"), "Brazil": ("BRL", "BR", "BRA"), "Canada": ("CAD", "CA", "CAN"),
         "Czechia": ("CZK", "CZ", "CZE"), "Euro Area": ("EUR", "XM", "EURO"), "Hungary": ("HUF", "HU", "HUN"),
         "India": ("INR", "IN", "IND"), "Indonesia": ("IDR", "ID", "IDN"), "Israel": ("ILS", "IL", "ISR"),
         "Japan": ("JPY", "JP", "JPN"), "Korea": ("KRW", "KR", "KOR"), "Mexico": ("MXN", "MX", "MEX"),
         "New Zealand": ("NZD", "NZ", "NZL"), "Norway": ("NOK", "NO", "NOR"), "Poland": ("PLN", "PL", "POL"),
         "South Africa": ("ZAR", "ZA", "ZAF"), "Sweden": ("SEK", "SE", "SWE"), "Switzerland": ("CHF", "CH", "CHE"),
         "Thailand": ("THB", "TH", "THA"), "UK": ("GBP", "GB", "GBR")}
CUR, BIS, ISO = ({c: v[i] for c, v in CODES.items()} for i in range(3))
EA = ["AUT", "BEL", "FIN", "FRA", "DEU", "IRL", "ITA", "NLD", "PRT", "ESP", "GRC", "SVN", "SVK", "LTU", "LVA"]
WEO_INDS = ["GGXWDG_NGDP", "NGDP_RPCH", "BCA_NGDPD", "GGXONLB_G01_GDP_PT"]   # debt, growth, current account, primary balance


# cached download for the pinned vintage
def fetch(url, name):
    path = os.path.join(CACHE, f"{name}_{VINTAGE}")
    if not os.path.exists(path):
        if VINTAGE != dt.date.today().isoformat():
            sys.exit(f"no cached {name} for vintage {VINTAGE}; set SILO1_VINTAGE={dt.date.today().isoformat()} to refresh")
        r = subprocess.run(["curl", "-s", "-f", "-m", "120", "--retry", "5", "--retry-delay", "3", "--retry-all-errors", "-o", path, url])
        if r.returncode != 0:
            sys.exit(f"download failed: {url}")
    return path


# panel and frozen model
df = core.load_raw()
X = ["D"] + core.CONTROLS
d = core.load()
beta = core.fit(d, "delta_E", X)["beta"]
means = d.groupby("country")[["delta_E"] + X].mean()
alpha = {c: float(means.loc[c, "delta_E"] - means.loc[c, X].to_numpy() @ beta) for c in means.index}
xbar = {c: means.loc[c, X].to_dict() for c in means.index}
ybar = {c: float(means.loc[c, "delta_E"]) for c in means.index}
p25 = df[df.year == 2025].set_index("country")

# WEO snapshot
WEO = {v: {k: json.load(open(os.path.join(CACHE, f"weo_{v}_{k}.json")))["values"][k] for k in WEO_INDS}
       for v in ("oct2026", "apr2026")}


# October 2026 series, April 2026 before October's start
def weo_series(ind, iso):
    s = {int(y): float(v) for y, v in WEO["oct2026"][ind][iso].items()}
    a = {int(y): float(v) for y, v in WEO["apr2026"][ind].get(iso, {}).items()}
    first = min(s)
    early = {y: a[y] for y in range(1999, first) if y in a}
    if early and abs(s[first] - a.get(first, np.inf)) > 0.25:
        sys.exit(f"{iso} {ind}: October starts {first}; April differs at the join")
    return pd.Series({**early, **s}).reindex(range(1999, 2027))


# 2026 WEO-derived columns
def weo_2026(c):
    debt, grow, ca, pb = (weo_series(ind, ISO[c]) for ind in WEO_INDS)
    step = debt.loc[2025] - debt.loc[2024]
    streak = 0                                    # primary-balance streak
    for x in pb.diff().loc[:2025]:
        streak = 0 if pd.isna(x) else (streak + 1 if x < 0 else 0)
    r6 = lambda v: round(float(v), 6)
    return dict(debt_gdp_lag=r6(debt.loc[2025]), gdp_growth=r6(grow.loc[2026]), ca_gdp=r6(ca.loc[2026]),
                prim_bal_lag=r6(pb.loc[2025]), delta_debt_pos_lag=r6(max(step, 0)), delta_debt_neg_lag=r6(min(step, 0)),
                prim_bal_streak_lag=float(streak))


# downloads
ecb = pd.read_csv(fetch("https://data-api.ecb.europa.eu/service/data/EXR/D." +
                        "+".join(sorted(set(CUR.values()) - {"EUR"} | {"USD"})) +
                        ".EUR.SP00.A?startPeriod=2024-12-01&format=csvdata", "ecb_exr.csv"))
fx = ecb.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE").sort_index()
fx.index = pd.to_datetime(fx.index)
usd = pd.DataFrame({c: (1.0 / fx["USD"] if cur == "EUR" else fx[cur] / fx["USD"])
                    for c, cur in CUR.items()}).dropna()        # local currency per USD
def level_on(day):
    return usd[usd.index <= pd.Timestamp(day)].iloc[-1], usd[usd.index <= pd.Timestamp(day)].index[-1]
lv25, d25 = level_on("2025-12-31"); lvT, dT = level_on(CUTOFF)
elapsed = (dT - pd.Timestamp("2025-12-31")).days / 365.0
realized = 100 * np.log(lvT / lv25)

fred = {}
for sid in ["VIXCLS", "FEDFUNDS"] + ["WUI" + ISO[c] for c in CUR if c != "Euro Area"] + ["WUI" + e for e in EA]:
    f = pd.read_csv(fetch(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", f"fred_{sid}.csv"))
    f.columns = ["date", "value"]; f["date"] = pd.to_datetime(f["date"])
    f["value"] = pd.to_numeric(f["value"], errors="coerce"); fred[sid] = f.dropna()
def ymean(sid, year):
    f = fred[sid]; f = f[(f.date.dt.year == year) & (f.date <= pd.Timestamp(CUTOFF))]
    return float(f.value.mean())
vix26, ff26 = ymean("VIXCLS", 2026), ymean("FEDFUNDS", 2026)
def wui(c, year):
    if c == "Euro Area":
        return float(np.mean([ymean("WUI" + e, year) for e in EA]))
    return ymean("WUI" + ISO[c], year)

bis = pd.read_csv(fetch("https://stats.bis.org/api/v1/data/WS_CBPOL/M." + "+".join(list(BIS.values()) + ["US"]) +
                        "/all?startPeriod=2025-12&format=csv", "bis_cbpol.csv"))
bis = bis[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].dropna()
bis = bis[bis.TIME_PERIOD <= CUTOFF[:7]]
# policy rate in a month, or the latest before the cutoff
def pol(area, month=None):
    b = bis[bis.REF_AREA == area].sort_values("TIME_PERIOD")
    return float(b[b.TIME_PERIOD == month].OBS_VALUE.iloc[0]) if month else float(b.OBS_VALUE.iloc[-1])

pcpi = json.load(open(fetch("https://www.imf.org/external/datamapper/api/v1/PCPIPCH", "imf_pcpipch.json")))["values"]["PCPIPCH"]
def weo_infl(c, year):
    return float(pcpi[ISO[c]][str(year)])

# 2026 regressors
rows = {}
us_now, us_dec = pol("US"), pol("US", "2025-12")
for c in CUR:
    w = weo_2026(c)
    own_now, own_dec = pol(BIS[c]), pol(BIS[c], "2025-12")
    r = dict(D=w["debt_gdp_lag"] / 100.0, i=ff26, vix=vix26,
             rate_spread=float(p25.loc[c, "rate_spread"]) + ((own_now - us_now) - (own_dec - us_dec)),
             gdp_growth=w["gdp_growth"], ca_gdp=w["ca_gdp"],
             infl_diff=float(p25.loc[c, "infl_diff"]) + ((weo_infl(c, 2026) - float(pcpi["USA"]["2026"]))
                                                          - (weo_infl(c, 2025) - float(pcpi["USA"]["2025"]))),
             wui=float(p25.loc[c, "wui"]) + (wui(c, 2026) - wui(c, 2025)),
             prim_bal_lag=w["prim_bal_lag"], delta_debt_pos_lag=w["delta_debt_pos_lag"],
             delta_debt_neg_lag=w["delta_debt_neg_lag"], prim_bal_streak_lag=w["prim_bal_streak_lag"])
    assert abs(r["D"] * 100 - float(p25.loc[c, "debt_gdp"])) < 1e-6   # the panel's vintage
    rows[c] = r

# forecasts and evaluation
res = {}
for c, r in rows.items():
    x = np.array([r[k] for k in X])
    full = alpha[c] + x @ beta
    contrib = {k: float(b * (r[k] - xbar[c][k])) for k, b in zip(X, beta)}   # against own 2001-2025 mean
    res[c] = dict(predicted_full_year=float(full), predicted_to_date=float(full * elapsed),
                  realized_to_date=float(realized[c]), contributions_full_year=contrib,
                  regressors_2026={k: float(v) for k, v in r.items()},
                  level_2025_12_31=float(lv25[c]), level_latest=float(lvT[c]))
real = np.array([res[c]["realized_to_date"] for c in CUR])
pred = np.array([res[c]["predicted_to_date"] for c in CUR])
hist = np.array([ybar[c] * elapsed for c in CUR])
rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
summary = dict(rmse_model=rmse(real - pred), rmse_no_change=rmse(real), rmse_hist_mean=rmse(real - hist), n=len(real))
krw = usd["Korea"][:CUTOFF]
won_path = dict(peak_2026=float(krw["2026-01-01":].max()), peak_2026_date=str(krw["2026-01-01":].idxmax().date()))
con = {c: r["contributions_full_year"] for c, r in res.items()}
rank = {c: 1 + sum(abs(v) > abs(con[c]["D"]) for n, v in con[c].items() if n != "D") for c in con}
cross = dict(direction_right=int(sum(np.sign(res[c]["predicted_to_date"]) == np.sign(res[c]["realized_to_date"]) for c in res)),
             median_abs_debt_part=float(np.median([abs(con[c]["D"] * elapsed) for c in con])),
             median_abs_actual=float(np.median([abs(res[c]["realized_to_date"]) for c in res])),
             debt_largest=int(sum(r == 1 for r in rank.values())), debt_median_rank=float(np.median(list(rank.values()))),
             n_inputs=len(X))
out = dict(vintage=VINTAGE, cutoff=CUTOFF, cross=cross, won_path=won_path, fx_dates=dict(start=str(d25.date()), end=str(dT.date())),
           share_of_year=elapsed, summary=summary, economies=res)
core.save("oos2026.json", out)
k = res["Korea"]
print(f"window {d25.date()} -> {dT.date()} ({elapsed:.3f} of a year)")
print("summary:", json.dumps(summary, indent=1))
print(f"Korea: realized {k['realized_to_date']:+.2f}%, predicted to date {k['predicted_to_date']:+.2f}% "
      f"(full year {k['predicted_full_year']:+.2f}%), debt part (full year) {k['contributions_full_year']['D']:+.2f}")
