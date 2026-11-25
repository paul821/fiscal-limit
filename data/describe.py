import os

import numpy as np
import pandas as pd

import core as core

d = core.load()
raw = core.load_raw()
est = core.read("estimates.json")
gamma, slope = est["H2"]["gamma"], est["H1"]["coef"]
out = {}

# Table 1
t1 = d.groupby("country").agg(n=("delta_E", "size"), mean_debt=("debt_gdp", "mean"), max_debt=("debt_gdp", "max"),
                              mean_dep=("delta_E", "mean"), sd_dep=("delta_E", "std")).reset_index()
t1["group"] = np.where(t1.country.isin(core.IMF_ADV), "Advanced", "Emerging")
out["table_countries"] = t1.to_dict(orient="records")

# Table 3
q = lambda s: dict(mean=float(s.mean()), sd=float(s.std()), p10=float(s.quantile(.1)), median=float(s.median()),
                   p90=float(s.quantile(.9)))
out["table_variables"] = {lab: q(d[col]) for lab, col in [
    ("Depreciation vs USD", "delta_E"), ("Gross debt/GDP (lagged)", "debt_gdp_lag"), ("Shadow short rate", "i"),
    ("Interest differential", "rate_spread"), ("Real GDP growth", "gdp_growth"), ("Current account/GDP", "ca_gdp"),
    ("Inflation differential", "infl_diff"), ("Primary balance (lagged)", "prim_bal_lag"), ("Debt increase (lagged)", "delta_debt_pos_lag"),
    ("Debt decrease (lagged)", "delta_debt_neg_lag"), ("Primary-balance streak (lagged)", "prim_bal_streak_lag"), ("VIX", "vix"),
    ("World Uncertainty Index", "wui"), ("Stress probability", "pi")]}

# Section 5: across economies
g = d.groupby("country").agg(debt=("debt_gdp", "mean"), dep=("delta_E", "mean")).reset_index()
adv = g.country.isin(core.IMF_ADV)
no_jp = g[adv & (g.country != "Japan")]
out["across"] = dict(corr=float(g.debt.corr(g.dep)), mean_dep_advanced=float(g[adv].dep.mean()),
                     mean_dep_emerging=float(g[~adv].dep.mean()), mean_debt_advanced=float(g[adv].debt.mean()),
                     mean_debt_emerging=float(g[~adv].debt.mean()),
                     japan_mean_dep=float(g[g.country == "Japan"].dep.iloc[0]),
                     n_advanced=int(adv.sum()), n_emerging=int((~adv).sum()),
                     mean_debt_advanced_no_japan=float(no_jp.debt.mean()), mean_dep_advanced_no_japan=float(no_jp.dep.mean()))

# Section 5: within economies
dm = d.D - d.groupby("country").D.transform("mean")
ss = (dm ** 2).groupby(d.country).sum()
share = (ss / ss.sum()).sort_values(ascending=False)
yearly_pi = raw.groupby("year").pi.first()
stress_years = [int(y) for y in yearly_pi[yearly_pi > core.STRESS_CUT].index]
p = raw.sort_values(["country", "year"]).copy()
p["ddebt"] = p.debt_gdp - p.debt_gdp_lag   # lag covers 2001
out["within"] = dict(
    share_by_economy={k: float(v) for k, v in share.items()},
    stress_years=stress_years, n_stress_years_in_sample=int(d[d.HS > 0].year.nunique()),
    debt_step_stressed=float(p[p.year.isin(stress_years)].ddebt.mean()),       # full panel, current debt
    debt_step_calm=float(p[~p.year.isin(stress_years) & p.ddebt.notna()].ddebt.mean()),
    mean_lagged_debt_stressed=float(d[d.HS > 0].debt_gdp_lag.mean()),        # estimation sample
    mean_lagged_debt_calm=float(d[d.HS == 0].debt_gdp_lag.mean()))


def crossings(thr):
    above = d.assign(above=d.D > thr)
    per = above.groupby("country").above.agg(["sum", "count"])
    sw = above.groupby("country").above.apply(lambda s: int((s.astype(int).diff().abs() == 1).sum()))
    return dict(gamma=float(thr), crossings=int(sw.sum()), economies_crossing=int((sw > 0).sum()),
                always_above=int((per["sum"] == per["count"]).sum()), always_below=int((per["sum"] == 0).sum()))


rows = [crossings(x) for x in core.grid(d)]
cr = np.array([r["crossings"] for r in rows]); ec = np.array([r["economies_crossing"] for r in rows])
out["crossings"] = dict(min=int(cr.min()), median=float(np.median(cr)), max=int(cr.max()),
                        economies_min=int(ec.min()), economies_max=int(ec.max()), at_estimated_break=crossings(gamma))

# facts
fred = lambda s: pd.read_csv(os.path.join(core.SRC, s + ".csv"), index_col=0, parse_dates=True,
                             na_values=".").iloc[:, 0].dropna()
k, j, u, se, no = fred("DEXKOUS"), fred("DEXJPUS"), fred("DEXUSEU"), fred("DEXSDUS"), fred("DEXNOUS")
avg = k[:"2025"].groupby(k[:"2025"].index.year).mean(); dec = k["2024-12"]; r26 = k["2026"]
jmax = j["2024-07"].max(); before = j[:"2024-06"]
ko = raw[raw.country == "Korea"].set_index("year")
bis = pd.read_csv(os.path.join(core.SRC, "bis_fx_turnover_shares.csv"), index_col=0)
out["facts"] = dict(
    won=dict(avg_2025=float(avg[2025]), avg_1998=float(avg[1998]), weakest_full_year=int(avg.idxmax()),
             first_above_1470_dec2024=str(dec[dec >= 1470].index[0].date()), first_above_1500_2026=str(r26[r26 >= 1500].index[0].date()),
             peak_2026=float(r26.max()), peak_2026_date=str(r26.idxmax().date())),
    yen=dict(jul2024_max=float(jmax), last_weaker_before=str(before[before >= jmax].index[-1].date())),
    euro=dict(jan2021_high=float(u["2021"].max()), sep2022_low=float(u["2022"].min()),
              fall_pct=float(100 * (u["2022"].min() / u["2021"].max() - 1))),
    krona=dict(record_usd=float(se.max()), record_date=str(se.idxmax().date())),
    krone=dict(record_usd=float(no.max()), record_date=str(no.idxmax().date())),
    n_debt_higher_2019_25=int((raw[raw.year >= 2019].groupby("country").debt_gdp.mean()
                               > raw[raw.year <= 2018].groupby("country").debt_gdp.mean()).sum()),
    mean_debt_2007=float(raw[raw.year == 2007].debt_gdp.mean()), mean_debt_2025=float(raw[raw.year == 2025].debt_gdp.mean()),
    korea=dict(debt_2018=float(ko.loc[2018, "debt_gdp"]), debt_2025=float(ko.loc[2025, "debt_gdp"]),
               implied_extra_depreciation_pp=float(slope * (ko.loc[2025, "debt_gdp"] - ko.loc[2018, "debt_gdp"]) / 100)),
    japan_2001=dict(lagged_debt=float(d[(d.country == "Japan") & (d.year == 2001)].debt_gdp_lag.iloc[0]),
                    depreciation=float(d[(d.country == "Japan") & (d.year == 2001)].delta_E.iloc[0]),
                    lagged_debt_min=float(d[d.country == "Japan"].debt_gdp_lag.min()),
                    lagged_debt_max=float(d[d.country == "Japan"].debt_gdp_lag.max())),
    bis_avg_share_of_200={c: float(bis.loc[c, "avg"]) for c in ("NZD", "HUF", "CZK", "ILS", "IDR", "THB", "CLP", "RUB")})

core.save("describe.json", out)
print("crossings over the grid:", out["crossings"]["min"], "to", out["crossings"]["max"], "| at the break:", out["crossings"]["at_estimated_break"])
