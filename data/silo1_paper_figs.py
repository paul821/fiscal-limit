import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import paperstyle
import silo1_core as core

paperstyle.use()
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.titlelocation": "center",
                     "axes.grid": False, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "lines.linewidth": 1.5, "lines.markersize": 3,
                     "axes.edgecolor": "#222222", "xtick.direction": "out", "ytick.direction": "out"})
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
CALM, STRESS, GREEN = "#0072B2", "#D55E00", "#009E73"
GRAY, INK, BAND = "#8A8F98", "#222222", "#E4E4E4"
DASH = (0, (4, 2))
EST, ROB, DES, OOS = core.read("estimates.json"), core.read("robustness.json"), core.read("describe.json"), core.read("oos2026.json")
CONTROLS = core.CONTROLS
df, d = core.load_raw(), core.load()
ADV = set(core.IMF_ADV)
CODE = {"Australia": "AUS", "Brazil": "BRA", "Canada": "CAN", "Czechia": "CZE", "Euro Area": "EUR",
        "Hungary": "HUN", "India": "IND", "Indonesia": "IDN", "Israel": "ISR", "Japan": "JPN",
        "Korea": "KOR", "Mexico": "MEX", "New Zealand": "NZL", "Norway": "NOR", "Poland": "POL", "Russia": "RUS",
        "South Africa": "ZAF", "Sweden": "SWE", "Switzerland": "CHE", "Thailand": "THA", "UK": "GBR"}


def fit(frame, xc, y="delta_E"):
    """core.fit, returned as coefficient Series and covariance DataFrame for plotting."""
    r = core.fit(frame, y, xc)
    return pd.Series(r["beta"], xc), pd.DataFrame(r["V"], xc, xc)


def line(ax, slope, var, x0=0, x1=150, **kw):
    """Draw slope*D from x0 to x1 percent of GDP, with a 95 percent band."""
    x = np.array([x0, x1], float); col = kw.get("color", CALM)
    ax.plot(x, slope * x / 100, **kw)
    if var is not None:
        hw = 1.96 * np.sqrt(var) * x / 100
        ax.fill_between(x, slope * x / 100 - hw, slope * x / 100 + hw, color=col, alpha=.14, lw=0)


def kink(ax, b0, bN, V, gamma, x1, **kw):
    """Draw b0*D + bN*(D - gamma)*1{D > gamma} from 0 to x1 percent of GDP, with a 95 percent band
    from the covariance V of (b0, bN) when V is given."""
    x = np.union1d(np.linspace(0, x1, 301), [gamma * 100]); x = x[x <= x1]
    D = x / 100; H = np.clip(D - gamma, 0, None); y = b0 * D + bN * H
    ax.plot(x, y, **kw)
    if V is not None:
        hw = 1.96 * np.sqrt(V[0, 0] * D ** 2 + V[1, 1] * H ** 2 + 2 * V[0, 1] * D * H)
        ax.fill_between(x, y - hw, y + hw, color=kw.get("color", CALM), alpha=.14, lw=0)


def piece(ax, fr, gamma, top, **kw):
    """Kinked fit at a given break, refit here so the band and the slopes agree."""
    b, V = fit(fr.assign(Dhi=(fr.D - gamma) * (fr.D > gamma)), ["D", "Dhi"] + CONTROLS)
    kink(ax, b["D"], b["Dhi"], V.loc[["D", "Dhi"], ["D", "Dhi"]].to_numpy(), gamma, top, **kw)
    return b["D"], b["D"] + b["Dhi"]


def split(fr):
    f = fr.assign(DHS=fr.D * fr.HS); b, V = fit(f, ["D", "DHS", "HS"] + CONTROLS)
    return (b["D"], V.loc["D", "D"]), (b["D"] + b["DHS"], V.loc["D", "D"] + V.loc["DHS", "DHS"] + 2 * V.loc["D", "DHS"])


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"paper_fig_{name}.{ext}"))
    plt.close(fig)


def legend_below(fig, handles, labels, bottom=.11, **tight):
    """Series names sit in one frameless row under the plot area."""
    fig.legend(handles, labels, loc="lower center", ncol=len(handles), frameon=False, fontsize=7,
               handlelength=2.6, columnspacing=1.5)
    fig.tight_layout(rect=(0, bottom, 1, 1), **tight)


def shape_axes(ax, ylabel=False, ymax=None, ymin=0):
    ax.set_xlim(0, 150); ax.set_xticks([0, 50, 100, 150]); ax.set_xlabel("Lagged debt (percent of GDP)", fontsize=6.8)
    ax.set_ylim(ymin, ymax)
    if ylabel:
        ax.set_ylabel("Depreciation from debt (points)")


def close(a, b, what):
    assert abs(a - b) < 1e-3, (what, a, b)


# ---------------------------------------------------------------- EDA: across economies
g = d.groupby("country").agg(debt=("debt_gdp", "mean"), dep=("delta_E", "mean")).reset_index()
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.axhline(0, color=GRAY, lw=0.6)
for adv, col, mk, lab in ((True, INK, "o", "IMF advanced"), (False, GREEN, "s", "IMF emerging")):
    s = g[g.country.isin(ADV) == adv]
    ax.scatter(s.debt, s.dep, color=col, marker=mk, s=16, zorder=3, label=lab)
OFF = {"NOR": (3, 3), "KOR": (-3, 4), "SWE": (3, 3), "AUS": (-5, -2), "POL": (3, 3), "NZL": (3, -8),
       "THA": (3, -8), "GBR": (3, 3), "HUN": (-3, 3), "IDN": (-3, 4)}
for _, r in g.iterrows():
    c = CODE[r.country]; o = OFF.get(c, (3, 3))
    ax.annotate(c, (r.debt, r.dep), xytext=o, textcoords="offset points", fontsize=5.8, color="#444",
                ha="right" if o[0] < 0 else "left")
ax.set_xlabel("Average gross debt (percent of GDP)")
ax.set_ylabel("Average depreciation (percent a year)")
ax.set_xlim(0, 200)
legend_below(fig, ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
save(fig, "eda_cross")

# ---------------------------------------------------------------- EDA: within economies
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for y in DES["within"]["stress_years"]:
    ax.axvspan(y - .5, y + .5, color=BAND, lw=0)
for c, s in df.sort_values("year").groupby("country"):
    if c not in ("Japan", "UK"):
        ax.plot(s.year, s.debt_gdp_lag, color="#B8BCC4", lw=.7)
for c, col, ls in (("Japan", INK, "-"), ("UK", CALM, "-")):
    s = df[df.country == c].sort_values("year").dropna(subset=["debt_gdp_lag"])
    ax.plot(s.year, s.debt_gdp_lag, color=col, ls=ls, lw=1.5)
gam = EST["H2"]["gamma"] * 100
ax.axhline(gam, color=INK, ls=":", lw=.9)
ax.set_xlim(2000.5, 2025.5); ax.set_ylim(0, 240); ax.set_xticks(range(2001, 2026, 4))
ax.set_ylabel("Lagged gross debt (percent of GDP)")
legend_below(fig, [Line2D([], [], color=INK, lw=1.5), Line2D([], [], color=CALM, lw=1.5), Line2D([], [], color="#B8BCC4", lw=.8),
                   Line2D([], [], color=INK, ls=":", lw=.9), Patch(color=BAND)],
             ["Japan", "United Kingdom", "Other economies", "Estimated threshold", "Stressed years"])
save(fig, "eda_paths")

# ---------------------------------------------------------------- four tests as estimated shapes
base = EST["H1"]["coef"]; thr = EST["H2"]; PR = EST["H4"]
g2 = thr["gamma"]; t0, tN, gC, gS = PR["theta0"], PR["thetaN"], PR["gamma_calm"], PR["gamma_stress"]
b1, V1 = fit(d, ["D"] + CONTROLS); close(b1["D"], base, "H1")
(sc, vc), (ss, vs) = split(d); close(sc, EST["H3"]["slope_calm"], "calm"); close(ss, EST["H3"]["slope_stressed"], "stressed")
fig, axs = plt.subplots(1, 4, figsize=(6.6, 2.5), sharey=True)
for a, t in zip(axs, ("H1: Linear", "H2: Threshold", "H3: Salience", "H4: Moving limit")):
    shape_axes(a, a is axs[0], 26, -3); a.set_title(t)
line(axs[0], base, V1.loc["D", "D"], color=CALM)
s1 = .12; s0 = s1 / 2.5   # sketch, drawn to no estimate: slopes in points per pp, ratio as in Figure 1
axs[1].plot([0, g2 * 100, 150], [0, s0 * g2 * 100, s0 * g2 * 100 + s1 * (150 - g2 * 100)], color=GRAY, ls=DASH, lw=.9)
lo, hi = piece(axs[1], d, g2, 150, color=CALM); close(lo, thr["slope_below"], "below"); close(hi, thr["slope_above"], "above")
axs[1].axvline(g2 * 100, color=INK, ls=":", lw=.7)
axs[2].plot([0, 135], [0, 15.3], color=GRAY, ls=DASH, lw=.9)
line(axs[2], sc, vc, color=CALM); line(axs[2], ss, vs, color=STRESS, ls=DASH)
th4 = np.where(d.HS > .5, gS, gC); fr = d.assign(Dhi=(d.D - th4) * (d.D > th4)); b4, V4 = fit(fr, ["D", "Dhi"] + CONTROLS)
close(b4["D"], t0, "H4 t0"); close(b4["Dhi"], tN, "H4 tN")
kink(axs[3], t0, tN, V4.loc[["D", "Dhi"], ["D", "Dhi"]].to_numpy(), gC, 150, color=CALM)
kink(axs[3], t0, tN, None, gS, 150, color=STRESS, ls=DASH)
axs[3].axvline(gC * 100, color=CALM, ls=":", lw=.8); axs[3].axvline(gS * 100, color=STRESS, ls=":", lw=.8)
legend_below(fig, [Line2D([], [], color=CALM, lw=1.5), Line2D([], [], color=STRESS, ls=DASH, lw=1.5), Line2D([], [], color=GRAY, ls=DASH, lw=.9),
                   Line2D([], [], color=INK, ls=":", lw=.8)],
             ["All or calm years", "Stressed years", "Shape the hypothesis predicts", "Estimated threshold"], bottom=.14, w_pad=.5)
save(fig, "shapes")

# ---------------------------------------------------------------- H1: Japan 2001
jp = d[d.country == "Japan"]
inf = ROB["influence"][0]; assert (inf["country"], inf["year"]) == ("Japan", 2001), inf   # panel (b) labels it
j_drop = ROB["slope_without_each_economy"]["Japan"]
nj = ROB["samples"]["no_influential"]["H1"]["coef"]
nojp = d[d.country != "Japan"]; top_nj = nojp.debt_gdp_lag.max()
fig, (a, b) = plt.subplots(1, 2, figsize=(6.4, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
o = jp[jp.year != 2001]; r = jp[jp.year == 2001].iloc[0]
a.scatter(o.debt_gdp_lag, o.delta_E, color=GRAY, s=12, zorder=3, label="Other years")
a.scatter([r.debt_gdp_lag], [r.delta_E], color=STRESS, marker="s", s=24, zorder=4, label="2001")
a.axhline(0, color=GRAY, lw=.6)
a.set_xlabel("Lagged debt (percent of GDP)"); a.set_ylabel("Yen depreciation (percent)"); a.set_title("(a) Japan")
a.legend(loc="lower right", frameon=False, fontsize=6.8)
line(b, base, V1.loc["D", "D"], color=CALM, label="All data")
b.plot([0, 150], [0, nj * 1.5], color=GREEN, label="Without Japan 2001")
b.plot([0, top_nj], [0, j_drop * top_nj / 100], color=INK, ls=DASH, label="Without Japan")
b.set_xlim(0, 150); b.set_ylim(0, 16.5); b.set_xticks([0, 50, 100, 150])
b.set_xlabel("Lagged debt (percent of GDP)"); b.set_ylabel("Depreciation from debt (points)"); b.set_title("(b) Fitted debt slope")
b.legend(loc="upper left", frameon=False, fontsize=6.8)
fig.tight_layout()
save(fig, "h1_japan")

# ---------------------------------------------------------------- H2: groups
comp = ROB["groups"]
groups = [(f"{t} ({len(m)})", k, set(m)) for t, k, m in
          [("IMF advanced", "imf_advanced", core.IMF_ADV), ("IMF emerging", "imf_emerging", core.IMF_EME)]]
fig, axs = plt.subplots(1, len(groups), figsize=(4.6, 2.6), sharey=True)
for a, (t, k, mem) in zip(axs, groups):
    r = comp[k]; gx = r["gamma"]; sub = d[d.country.isin(mem)]
    top = min(150, sub.debt_gdp_lag.max()); th0 = r["linear_slope"]
    shape_axes(a, a is axs[0], 35, -8); a.set_title(t, fontsize=7.6)
    a.plot([0, top], [0, th0 * top / 100], color=GRAY, ls=DASH, lw=.9)
    lo, hi = piece(a, sub, gx, top, color=CALM); close(lo, r["slope_below"], k); close(hi, r["slope_above"], k)
    a.axvline(gx * 100, color=INK, ls=":", lw=.7)
legend_below(fig, [Line2D([], [], color=CALM, lw=1.5), Line2D([], [], color=GRAY, ls=DASH, lw=.9), Line2D([], [], color=INK, ls=":", lw=.8)],
             ["Threshold fit", "Linear fit", "Estimated threshold"], bottom=.13, w_pad=.5)
save(fig, "h2_groups")

# ---------------------------------------------------------------- H3: stress measures
def around_mean(cond):
    f = d.dropna(subset=[cond]).copy(); z = (f[cond] - f[cond].mean()) / f[cond].std()
    f = f.assign(**{cond + "_z": z, "Dc": f.D * z})
    b, V = fit(f, ["D", "Dc", cond + "_z"] + [x for x in CONTROLS if x != cond])
    lo = (b["D"] - b["Dc"], V.loc["D", "D"] + V.loc["Dc", "Dc"] - 2 * V.loc["D", "Dc"])
    hi = (b["D"] + b["Dc"], V.loc["D", "D"] + V.loc["Dc", "Dc"] + 2 * V.loc["D", "Dc"])
    return lo, hi
panels = [("VIX regime", (sc, vc), (ss, vs))]
for t, cond in (("Global Economic Policy Uncertainty", "gepu"), ("World Uncertainty Index", "wui")):
    lo, hi = around_mean(cond); close(lo[0], ROB["text_measures"][cond]["slope_low"], cond); close(hi[0], ROB["text_measures"][cond]["slope_high"], cond)
    panels.append((t, lo, hi))
fig, axs = plt.subplots(1, 3, figsize=(6.4, 2.6), sharey=True)
for a, (t, s1, s2) in zip(axs, panels):
    shape_axes(a, a is axs[0], 21, -3); a.set_title(t, fontsize=7.6)
    line(a, s1[0], s1[1], color=CALM); line(a, s2[0], s2[1], color=STRESS, ls=DASH)
legend_below(fig, [Line2D([], [], color=CALM, lw=1.5), Line2D([], [], color=STRESS, ls=DASH, lw=1.5)],
             ["Calm or low uncertainty", "Stressed or high uncertainty"], bottom=.13, w_pad=.6)
save(fig, "h3_measures")

# ---------------------------------------------------------------- H4: timeline, one plot with two vertical axes
yr_pi = df.groupby("year").pi.first(); yr_debt = d.groupby("year").debt_gdp_lag.mean()
yr_pi = yr_pi[yr_pi.index.isin(yr_debt.index)]
fig, ax = plt.subplots(figsize=(5.4, 3.3))
ax2 = ax.twinx()
ax2.bar(yr_pi.index, yr_pi.values, color=[STRESS if v > .5 else "#F0CDB6" for v in yr_pi.values], width=.7, alpha=.75, lw=0)
ax2.set_ylim(0, 1.0); ax2.set_yticks([0, .25, .5, .75, 1]); ax2.set_ylabel(r"Stress probability $\hat{\pi}_t$")
ax2.spines["right"].set_visible(True); ax2.grid(False)
ax.plot(yr_debt.index, yr_debt.values, color=INK, lw=1.7, marker="o", ms=2.6)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_ylim(40, 80); ax.set_yticks(range(40, 81, 10)); ax.set_ylabel("Mean lagged debt (percent of GDP)")
ax.set_xlim(2000.3, 2025.7); ax.set_xticks(range(2001, 2026, 4))
legend_below(fig, [Line2D([], [], color=INK, lw=1.7, marker="o", ms=2.6), Patch(color=STRESS, alpha=.75), Patch(color="#F0CDB6", alpha=.75)],
             ["Mean lagged debt (left axis)", r"$\hat{\pi}_t>0.5$ (right axis)", r"$\hat{\pi}_t\leq 0.5$ (right axis)"])
save(fig, "h4_timeline")

# ---------------------------------------------------------------- Japan: H3
fig, axs = plt.subplots(1, 2, figsize=(5.0, 2.7), sharey=True)
for a, t, sample, h in ((axs[0], f"All {d.country.nunique()} economies", d, EST["H3"]),
                        (axs[1], "Without Japan", nojp, ROB["samples"]["no_japan"]["H3"])):
    top = 150 if a is axs[0] else top_nj
    (c0, cv), (s0, sv) = split(sample); close(c0, h["slope_calm"], t); close(s0, h["slope_stressed"], t)
    shape_axes(a, a is axs[0], 23, -3); a.set_title(t)
    line(a, c0, cv, 0, top, color=CALM); line(a, s0, sv, 0, top, color=STRESS, ls=DASH)
legend_below(fig, [Line2D([], [], color=CALM, lw=1.5), Line2D([], [], color=STRESS, ls=DASH, lw=1.5)], ["Calm years", "Stressed years"], bottom=.12)
save(fig, "japan_h3")

# ---------------------------------------------------------------- 2026: the won
e = pd.read_csv(os.path.join(HERE, "raw", "oos2026", f"ecb_exr.csv_{OOS['vintage']}"))
fx = e.pivot(index="TIME_PERIOD", columns="CURRENCY", values="OBS_VALUE"); fx.index = pd.to_datetime(fx.index)
krw = (fx["KRW"] / fx["USD"]).dropna(); krw = krw[(krw.index >= "2025-01-01") & (krw.index <= OOS["cutoff"])]
kor = OOS["economies"]["Korea"]; t_end = pd.Timestamp(OOS["fx_dates"]["end"]); t26 = pd.Timestamp("2025-12-31")
days = np.arange(0, (t_end - t26).days + 1)
fc = kor["level_2025_12_31"] * np.exp(kor["predicted_full_year"] / 100 * days / 365.0)
fig, ax = plt.subplots(figsize=(5.6, 3.1))
ax.axvline(pd.Timestamp("2026-01-01"), color=GRAY, ls=":", lw=.8)
ax.plot(krw.index, krw.values, color=INK, lw=1.1, label="Actual")
ts = [t26 + pd.Timedelta(days=int(x)) for x in days]
ax.plot(ts, fc, color=CALM, ls=DASH, lw=1.5, label="Model forecast")
ax.set_xlim(krw.index[0], t_end); ax.set_ylim(1300, 1600); ax.set_ylabel("Won per US dollar")
ax.set_xticks([pd.Timestamp(x) for x in ("2025-01-01", "2025-07-01", "2026-01-01", "2026-07-01")])
ax.set_xticklabels(["Jan 2025", "Jul 2025", "Jan 2026", "Jul 2026"])
h, l = ax.get_legend_handles_labels(); legend_below(fig, h, l)
save(fig, "oos_won")

# ---------------------------------------------------------------- 2026: every currency in the check
fig, ax = plt.subplots(figsize=(3.5, 3.6))
ax.axhline(0, color=GRAY, lw=.6); ax.axvline(0, color=GRAY, lw=.6)
ax.plot([-14, 14], [-14, 14], color=GRAY, ls=DASH, lw=.9)
oth = [r for c, r in OOS["economies"].items() if c != "Korea"]
ax.scatter([r["predicted_to_date"] for r in oth], [r["realized_to_date"] for r in oth], color=INK, s=13, zorder=3, label="Other economies")
ax.scatter(kor["predicted_to_date"], kor["realized_to_date"], color=STRESS, marker="s", s=26, zorder=4, label="Korea")
ax.set_xlim(-14, 14); ax.set_ylim(-14, 14); ax.set_aspect("equal")
ax.set_xlabel("Model forecast (percent)"); ax.set_ylabel("Actual change (percent)")
h, l = ax.get_legend_handles_labels(); legend_below(fig, h, l, bottom=.1)
save(fig, "oos_scatter")
