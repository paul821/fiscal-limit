import os

import numpy as np
import pandas as pd

import silo1_core as core

C = core.CONTROLS
d = core.load()
out = {}

# influence: the observation whose removal moves the linear slope most
lin = core.fit(d, "delta_E", ["D"] + C)
b_full = lin["b"]["D"]
chg = np.array([core.fit(d.drop(i), "delta_E", ["D"] + C)["b"]["D"] - b_full for i in range(len(d))])
order = np.argsort(-np.abs(chg))
out["influence"] = [dict(country=str(d.loc[i, "country"]), year=int(d.loc[i, "year"]), slope_change=float(chg[i]))
                    for i in order[:5]]
top = order[0]


def four_tests(frame, tag):
    ctr = core.controls_in(frame)
    f = frame.assign(DHS=frame.D * frame.HS, Dpi=frame.D * frame.pi)
    m = core.fit(f, "delta_E", ["D", "DHS", "HS"] + ctr)
    return dict(
        n=len(frame), economies=int(frame.country.nunique()),
        H1=core.wild_test(frame, "delta_E", ["D"] + ctr, "D", rng=core.stream(f"{tag} H1")),
        H2=core.threshold_test(frame, ctr, rng=core.stream(f"{tag} H2")),
        H3=dict(split=core.wild_test(f, "delta_E", ["D", "DHS", "HS"] + ctr, "DHS", rng=core.stream(f"{tag} H3 split")),
                continuous=core.wild_test(f, "delta_E", ["D", "Dpi", "pi"] + ctr, "Dpi", rng=core.stream(f"{tag} H3 continuous")),
                slope_calm=float(m["b"]["D"]), slope_stressed=float(m["b"]["D"] + m["b"]["DHS"])),
        H4=core.moving_limit_test(frame, ctr, rng=core.stream(f"{tag} H4")))


samples = {"no_influential": d.drop(top), "no_japan": d[d.country != "Japan"]}
out["samples"] = {k: four_tests(v.reset_index(drop=True), k) for k, v in samples.items()}
for k, v in out["samples"].items():
    print(f"{k:15s} H1 {v['H1']['coef']:.2f} p1 {v['H1']['p_wild_upper']:.4f} | H2 p {v['H2']['p']:.4f} | "
          f"H3 p1 {v['H3']['split']['p_wild_upper']:.4f} | H4 p {v['H4']['p']:.4f}", flush=True)

# groups
groups = {"imf_advanced": core.IMF_ADV, "imf_emerging": core.IMF_EME}
out["groups"] = {}
for k, members in groups.items():
    sub = d[d.country.isin(members)].reset_index(drop=True); ctr = core.controls_in(sub)
    out["groups"][k] = dict(core.threshold_test(sub, ctr, rng=core.stream(f"group {k}")),
                            linear_slope=float(core.fit(sub, "delta_E", ["D"] + ctr)["b"]["D"]),
                            economies=int(sub.country.nunique()))
    print(f"group {k:15s} break {out['groups'][k]['gamma']:.4f} p {out['groups'][k]['p']:.4f}", flush=True)

# linear slope checks
out["driscoll_kraay"] = core.driscoll_kraay(d, "delta_E", ["D"] + C, "D")
sp = pd.read_csv(os.path.join(core.SRC, "areaer_non_floating_spells.csv"))
mask = np.zeros(len(d), bool)
for _, r in sp.iterrows():
    mask |= ((d.country == r.country) & d.year.between(r.first_year, r.last_year)).to_numpy()
# the four tests without the economy-years the IMF classifies as non-floating
fl = d[~mask].reset_index(drop=True)
out["floating_sample"] = four_tests(fl, "floating")
out["floating_years"] = dict(out["floating_sample"]["H1"], dropped=int(mask.sum()))
# whether the linear slope differs between the IMF advanced and emerging groups (two-sided)
fa = d.assign(DA=d.D * d.country.isin(core.IMF_ADV))
mg = core.fit(fa, "delta_E", ["D", "DA"] + C)
out["group_slope"] = dict(core.wild_test(fa, "delta_E", ["D", "DA"] + C, "DA", rng=core.stream("group slope")),
                          slope_emerging=float(mg["b"]["D"]), slope_advanced=float(mg["b"]["D"] + mg["b"]["DA"]))
# H2 fit at every observed debt ratio between the 1st and 99th percentiles, beyond the trimmed grid
# (Section 6, Step 3): the fall in SSE against the linear model at the best candidate overall,
# below the grid, and within the grid's range
Dv = d.D.to_numpy(float)
cand = np.unique(Dv[(Dv >= np.quantile(Dv, .01)) & (Dv <= np.quantile(Dv, .99))])
gain = float((lin["u"] ** 2).sum()) - core.ThresholdDesign(d, C, [np.full(len(d), g) for g in cand]).rss(core._outcome(d))[:, 0]
G = core.grid(d)


def _best(mask):
    g = float(cand[mask][np.argmax(gain[mask])])
    return dict(gamma=g, sse_fall=float(gain[mask].max()), share_below=float((Dv <= g).mean()),
                n_above=int((Dv > g).sum()), economies_above=int(d.country[Dv > g].nunique()),
                n_below=int((Dv <= g).sum()), economies_below=int(d.country[Dv <= g].nunique()))


out["h2_full_scan"] = dict(overall=_best(cand > 0), below_grid=_best(cand < G.min()),
                           within_grid=_best((cand >= G.min()) & (cand <= G.max())))
# H3 with Japan kept and one stressed-year intercept shared by the two safe havens of Ranaldo and
# Soderlind (2010), so the franc, a haven at ordinary debt, measures how much a haven strengthens in stress
hv = d.country.isin(["Japan", "Switzerland"]).astype(float)
fh = d.assign(DHS=d.D * d.HS, HavHS=hv * d.HS, Dpi=d.D * d.pi, Havpi=hv * d.pi)
out["h3_haven"] = {}
for form, xs, term, haven in (("binary", ["D", "DHS", "HS", "HavHS"], "DHS", "HavHS"),
                              ("continuous", ["D", "Dpi", "pi", "Havpi"], "Dpi", "Havpi")):
    m = core.fit(fh, "delta_E", xs + C)
    out["h3_haven"][form] = dict(core.wild_test(fh, "delta_E", xs + C, term, rng=core.stream(f"H3 haven {form}")),
                                 slope_calm=float(m["b"]["D"]), slope_stressed=float(m["b"]["D"] + m["b"][term]),
                                 haven=float(m["b"][haven]), haven_se=float(m["se"][haven]),
                                 dk=core.driscoll_kraay(fh, "delta_E", xs + C, term))
    print(f"H3 haven {form:10s} shift {m['b'][term]:.3f}  p one-sided {out['h3_haven'][form]['p_wild_upper']:.4f}", flush=True)


def _dhs_shares(extra):
    """Each economy's share of the variation in D*HS left after the fixed effects, D, HS, the
    controls, and `extra`: the variation from which the binary H3 shift is estimated."""
    c, ng = core.codes(fh)
    Z = core.demean(fh[["D", "HS"] + extra + C].to_numpy(float), c, ng)
    x = core.demean(fh[["DHS"]].to_numpy(float), c, ng)[:, 0]
    r = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    s = pd.Series(r ** 2).groupby(fh.country.to_numpy()).sum()
    return (s / s.sum()).to_dict()


out["h3_leverage"] = dict(baseline=_dhs_shares([]), with_haven=_dhs_shares(["HavHS"]))

# The tests with year fixed effects, which absorb every shock common to all economies in a year,
# in place of the measured global regressors (shadow rate, VIX) and the stress level they would duplicate
yrs = sorted(d.year.unique())
YD = [f"y{y}" for y in yrs[1:]]
glob = [c for c in C if d.groupby("year")[c].nunique().max() == 1]
cy = [c for c in C if c not in glob] + YD
fy = fh.assign(**{f"y{y}": (fh.year == y).astype(float) for y in yrs[1:]})
my = core.fit(fy, "delta_E", ["D", "DHS"] + cy)
out["year_fe"] = dict(
    absorbed=glob,
    H1=core.wild_test(fy, "delta_E", ["D"] + cy, "D", rng=core.stream("year fe H1")),
    H2=core.threshold_test(fy, cy, rng=core.stream("year fe H2")),
    H3=dict(split=core.wild_test(fy, "delta_E", ["D", "DHS"] + cy, "DHS", rng=core.stream("year fe H3 split")),
            continuous=core.wild_test(fy, "delta_E", ["D", "Dpi"] + cy, "Dpi", rng=core.stream("year fe H3 continuous")),
            slope_calm=float(my["b"]["D"]), slope_stressed=float(my["b"]["D"] + my["b"]["DHS"])),
    haven=core.wild_test(fy, "delta_E", ["D", "DHS", "HavHS"] + cy, "DHS", rng=core.stream("year fe H3 haven")),
    H4=core.moving_limit_test(fy, cy, rng=core.stream("year fe H4")))
print("year fe", {k: round(v["p"] if "p" in v else v.get("p_wild_upper", v.get("split", {}).get("p_wild_upper", -1)), 4)
                  for k, v in out["year_fe"].items() if isinstance(v, dict)}, flush=True)
# Advanced economies' share of the economy-years above and below the full-panel threshold (Section 7.4)
g2 = core.read("estimates.json")["H2"]["gamma"]; adv = d.country.isin(core.IMF_ADV)
out["advanced_share_at_threshold"] = dict(gamma=g2, above=float(adv[d.D > g2].mean()), below=float(adv[d.D <= g2].mean()))
# Comovement of the linear model's residuals across economies, which the size simulation leaves out (Section 7.2)
res = pd.DataFrame(dict(country=d.country.to_numpy(), year=d.year.to_numpy(), u=lin["u"]))
cm = res.pivot(index="year", columns="country", values="u").corr(min_periods=10).to_numpy()
out["residual_comovement"] = dict(year_share=float((res.groupby("year").u.transform("mean") ** 2).sum() / (res.u ** 2).sum()),
                                  mean_pairwise_corr=float(np.nanmean(cm[np.triu_indices_from(cm, 1)])))
out["slope_without_each_economy"] = {c: float(core.fit(d[d.country != c], "delta_E", ["D"] + C)["b"]["D"])
                                     for c in sorted(d.country.unique())}

# H3 with text-based uncertainty indices (one standard deviation below and above average)
out["text_measures"] = {}
for cond in ("gepu", "wui"):
    f = d.dropna(subset=[cond]).copy()
    f = f.assign(**{cond + "_z": (f[cond] - f[cond].mean()) / f[cond].std()})
    f = f.assign(Dc=f.D * f[cond + "_z"])
    xs = ["D", "Dc", cond + "_z"] + [c for c in C if c != cond]
    r = core.wild_test(f, "delta_E", xs, "Dc", rng=core.stream(f"text {cond}"))
    m = core.fit(f, "delta_E", xs)
    out["text_measures"][cond] = dict(r, slope_low=float(m["b"]["D"] - m["b"]["Dc"]), slope_high=float(m["b"]["D"] + m["b"]["Dc"]))

# The same World Uncertainty Index test in the other samples of Table 6: without Japan's 2001
# economy-year, without Japan, and with Japan kept and the two havens' own response to the index
# (Haven x WUI), the counterpart of the shared stressed-year intercept. WUI is standardized within each sample.
out["wui_samples"] = {}
for tag, frame, haven in (("no_influential", samples["no_influential"], False), ("no_japan", samples["no_japan"], False),
                          ("floating", fl, False), ("safe_havens", d, True)):
    f = frame.dropna(subset=["wui"]).reset_index(drop=True)
    f = f.assign(wui_z=(f.wui - f.wui.mean()) / f.wui.std())
    f = f.assign(Dc=f.D * f.wui_z, Havz=f.country.isin(["Japan", "Switzerland"]).astype(float) * f.wui_z)
    xs = ["D", "Dc", "wui_z"] + (["Havz"] if haven else []) + [c for c in core.controls_in(f) if c != "wui"]
    r = core.wild_test(f, "delta_E", xs, "Dc", rng=core.stream(f"text wui {tag}"))
    m = core.fit(f, "delta_E", xs)
    out["wui_samples"][tag] = dict(r, slope_low=float(m["b"]["D"] - m["b"]["Dc"]), slope_high=float(m["b"]["D"] + m["b"]["Dc"]))
    print(f"WUI {tag:15s} high-low {2 * m['b']['Dc']:.2f}  p one-sided {r['p_wild_upper']:.4f}", flush=True)

core.save("robustness.json", out)
