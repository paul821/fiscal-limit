import json
import os
import zlib

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
SRC = os.path.join(HERE, "raw")
PANEL = os.path.join(HERE, "panel", "panel21.csv")   # built by build_panel.py

CONTROLS = ["i", "vix", "rate_spread", "gdp_growth", "ca_gdp", "infl_diff", "wui",
            "prim_bal_lag", "delta_debt_pos_lag", "delta_debt_neg_lag", "prim_bal_streak_lag"]
USED = ["delta_E", "D", "pi", "HS", "country", "year"] + CONTROLS
STRESS_CUT = 0.5                    # stressed year
GRID = (.15, .85, 25)               # 25 quantiles, 15th to 85th percentile
B = int(os.environ.get("SILO1_B", 9999))   # bootstrap replications
BASE_SEED = 20261110

IMF_ADV = ["Australia", "Canada", "Czechia", "Euro Area", "Israel", "Japan", "Korea",
           "New Zealand", "Norway", "Sweden", "Switzerland", "UK"]
IMF_EME = ["Brazil", "Hungary", "India", "Indonesia", "Mexico", "Poland", "Russia", "South Africa", "Thailand"]


# seeded generator per test name
def stream(name):
    return np.random.default_rng([BASE_SEED, zlib.crc32(name.encode())])


# panel with model names
def load_raw():
    df = pd.read_csv(PANEL)
    df["D"] = df["debt_gdp_lag"] / 100.0
    df["i"] = df["shadow_rate"]
    df["pi"] = df["pi_stress_annual"]
    df["HS"] = (df["pi"] > STRESS_CUT).astype(float)
    return df


# estimation sample
def load():
    return load_raw().dropna(subset=USED).sort_values(["country", "year"]).reset_index(drop=True)


# controls that vary in the sample
def controls_in(frame):
    return [c for c in CONTROLS if frame[c].std() > 1e-9]


def save(name, obj):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=float)


def read(name):
    return json.load(open(os.path.join(OUT, name)))


# within estimator
def codes(frame):
    c, _ = pd.factorize(frame["country"].values)
    return c, int(c.max() + 1)


def demean(mat, c, ng):
    sums = np.zeros((ng, mat.shape[1]))
    np.add.at(sums, c, mat)
    return mat - (sums / np.bincount(c, minlength=ng).astype(float)[:, None])[c]


# CR1 small-sample factor
def cr1(n, k, ng):
    return (ng / (ng - 1)) * ((n - 1) / (n - k))


def cluster_vcov(X, u, XtXi, c, ng):
    meat = np.zeros((X.shape[1],) * 2)
    for g in range(ng):
        s = X[c == g].T @ u[c == g]
        meat += np.outer(s, s)
    return cr1(len(u), X.shape[1], ng) * (XtXi @ meat @ XtXi)


# within OLS with economy effects
def fit(frame, y, xcols):
    c, ng = codes(frame)
    dm = demean(frame[[y] + xcols].to_numpy(float), c, ng)
    Y, X = dm[:, 0], dm[:, 1:]
    XtXi = np.linalg.pinv(X.T @ X)
    beta = XtXi @ (X.T @ Y)
    u = Y - X @ beta
    rss = float(u @ u)
    V = cluster_vcov(X, u, XtXi, c, ng)
    se = np.sqrt(np.diag(V))
    t = beta / se
    return dict(xcols=list(xcols), b=dict(zip(xcols, beta)), beta=beta, se=dict(zip(xcols, se)),
                t=dict(zip(xcols, t)), p=dict(zip(xcols, 2 * stats.t.sf(np.abs(t), ng - 1))),
                p_upper=dict(zip(xcols, stats.t.sf(t, ng - 1))), V=V, u=u, X=X, Y=Y, XtXi=XtXi,
                rss=rss, r2=1 - rss / float((Y - Y.mean()) @ (Y - Y.mean())), k=X.shape[1],
                codes=c, ng=ng, n=len(Y))


# bootstrap primitives
def _chunks(total, size=1000):
    done = 0
    while done < total:
        yield min(size, total - done)
        done += size


def _indicator(c, ng):
    M = np.zeros((ng, len(c)))
    M[c, np.arange(len(c))] = 1.0
    return M


# draws at least as extreme; exact ties count
def _at_least(stat, obs, tol=1e-9):
    return stat >= obs - tol * abs(obs)


WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


# null draws: Rademacher, or Webb six-point when 2^G < B
def _null_draws(fitted, u0, c, ng, R, rng):
    if 2 ** ng >= B:
        W = np.where(rng.random((R, ng)) < 0.5, 1.0, -1.0)
    else:
        W = WEBB[rng.integers(0, 6, (R, ng))]
    return fitted[:, None] + u0[:, None] * W.T[c]


# wild-cluster bootstrap-t, null imposed
def wild_test(frame, y, xfull, term, B=B, rng=None):
    full = fit(frame, y, xfull)
    j = full["xcols"].index(term)
    tobs = full["t"][term]
    r0 = fit(frame, y, [x for x in xfull if x != term])
    X, XtXi, c, ng = full["X"], full["XtXi"], full["codes"], full["ng"]
    a = X @ XtXi[:, j]                                   # observation weights in coef j
    M, k = _indicator(c, ng), cr1(full["n"], full["k"], ng)
    up = two = 0
    for R in _chunks(B):
        Ys = _null_draws(r0["Y"] - r0["u"], r0["u"], c, ng, R, rng)
        Bs = XtXi @ (X.T @ Ys)
        S = M @ (a[:, None] * (Ys - X @ Bs))
        t = Bs[j] / np.sqrt(k * (S ** 2).sum(0))
        two += int(_at_least(t ** 2, tobs ** 2).sum())
        up += int(_at_least(t, tobs).sum())
    return dict(coef=float(full["b"][term]), se=float(full["se"][term]), t=float(tobs),
                p_cluster=float(full["p"][term]), p_cluster_upper=float(full["p_upper"][term]),
                p_wild=(two + 1) / (B + 1), p_wild_upper=(up + 1) / (B + 1), B=B, n=full["n"])


# threshold machinery
def grid(frame, lo=GRID[0], hi=GRID[1], n=GRID[2]):
    return np.unique(np.round(np.quantile(frame.D, np.linspace(lo, hi, n)), 4))


# kink design over candidate thresholds
class ThresholdDesign:

    def __init__(self, frame, ctr, candidates):
        self.c, self.ng = codes(frame)
        self.n = len(frame)
        Dv = frame.D.to_numpy(float)
        self.base = demean(frame[["D"] + ctr].to_numpy(float), self.c, self.ng)
        self.H = demean(np.column_stack([(Dv - th) * (Dv > th) for th in candidates]), self.c, self.ng)
        self.k = self.base.shape[1] + 1
        self.XtXi = np.array([np.linalg.pinv(self.X(m).T @ self.X(m)) for m in range(self.H.shape[1])])
        self.M = _indicator(self.c, self.ng)
        self.cr = cr1(self.n, self.k, self.ng)

    def X(self, m):
        return np.column_stack([self.base[:, :1], self.H[:, m], self.base[:, 1:]])

    # SSE per candidate and draw
    def rss(self, Y):
        BY, HY, yy = self.base.T @ Y, self.H.T @ Y, (Y * Y).sum(0)
        out = np.empty((self.H.shape[1], Y.shape[1]))
        for m in range(self.H.shape[1]):
            XtY = np.vstack([BY[:1], HY[m:m + 1], BY[1:]])
            out[m] = yy - (XtY * (self.XtXi[m] @ XtY)).sum(0)
        return out

    # least-squares candidate per draw
    def search(self, Y):
        r = self.rss(Y)
        idx = np.argmin(r, axis=0)
        return idx, r[idx, np.arange(Y.shape[1])]

    # Wald on the slope shift per draw
    def shift_wald(self, Y, idx):
        w = np.empty(Y.shape[1]); b = np.empty((self.k, Y.shape[1]))
        for m in np.unique(idx):
            cols = idx == m
            X, XtXi = self.X(m), self.XtXi[m]
            bm = XtXi @ (X.T @ Y[:, cols])
            S = self.M @ ((X @ XtXi[:, 1])[:, None] * (Y[:, cols] - X @ bm))
            w[cols] = bm[1] ** 2 / (self.cr * (S ** 2).sum(0))
            b[:, cols] = bm
        return w, b


def _outcome(frame, y="delta_E"):
    c, ng = codes(frame)
    return demean(frame[[y]].to_numpy(float), c, ng)


def threshold_fit(frame, gamma, ctr=CONTROLS):
    return fit(frame.assign(Dhi=(frame.D - gamma) * (frame.D > gamma)), "delta_E", ["D", "Dhi"] + ctr)


# H2: Wald at the searched threshold
def threshold_test(frame, ctr=CONTROLS, B=B, rng=None):
    G = grid(frame)
    td = ThresholdDesign(frame, ctr, [np.full(len(frame), g) for g in G])
    Y = _outcome(frame)
    i_hat, _ = td.search(Y)
    w_obs, _ = td.shift_wald(Y, i_hat)
    gamma = float(G[i_hat[0]])
    out = dict(gamma=gamma, wald=float(w_obs[0]), grid_lo=float(G.min()), grid_hi=float(G.max()),
               n_grid=int(len(G)), B=B)
    lin = fit(frame, "delta_E", ["D"] + ctr)
    cnt = 0
    for R in _chunks(B):
        Ys = _null_draws(lin["Y"] - lin["u"], lin["u"], lin["codes"], lin["ng"], R, rng)
        idx, _ = td.search(Ys)
        cnt += int(_at_least(td.shift_wald(Ys, idx)[0], w_obs[0]).sum())
    out["p"] = (cnt + 1) / (B + 1)
    m = threshold_fit(frame, gamma, ctr)
    out.update(slope_below=float(m["b"]["D"]), slope_above=float(m["b"]["D"] + m["b"]["Dhi"]),
               shift=float(m["b"]["Dhi"]), shift_p_cluster=float(m["p"]["Dhi"]))
    return out


def _split_candidates(frame, G):
    hs = frame.HS.to_numpy(float) > STRESS_CUT
    pairs = [(gc, gs) for gc in G for gs in G]
    return pairs, [np.where(hs, gs, gc) for gc, gs in pairs]


# H4: F for calm and stressed thresholds
def moving_limit_test(frame, ctr=CONTROLS, B=B, rng=None):
    G = grid(frame); n = len(frame)
    single = ThresholdDesign(frame, ctr, [np.full(n, g) for g in G])
    pairs, cands = _split_candidates(frame, G)
    split = ThresholdDesign(frame, ctr, cands)
    Y = _outcome(frame)
    i1, rss1 = single.search(Y)
    i2, rss2 = split.search(Y)
    F = float((rss1[0] - rss2[0]) / (rss2[0] / n))
    g1 = float(G[i1[0]]); gc, gs = (float(x) for x in pairs[i2[0]])
    r1 = threshold_fit(frame, g1, ctr)
    Fstar = []
    for R in _chunks(B):
        Ys = _null_draws(r1["Y"] - r1["u"], r1["u"], r1["codes"], r1["ng"], R, rng)
        a = single.search(Ys)[1]; b = split.search(Ys)[1]
        Fstar.append((a - b) / (b / n))
    Fstar = np.concatenate(Fstar)
    th = np.where(frame.HS.to_numpy(float) > STRESS_CUT, gs, gc)
    r2 = fit(frame.assign(Dhi=(frame.D - th) * (frame.D > th)), "delta_E", ["D", "Dhi"] + ctr)
    return dict(gamma_calm=gc, gamma_stress=gs, inward_shift_pp=100 * (gc - gs), F=F,
                p=float((_at_least(Fstar, F).sum() + 1) / (B + 1)), F_crit_90=float(np.quantile(Fstar, .90)),
                theta0=float(r2["b"]["D"]), thetaN=float(r2["b"]["Dhi"]), B=B)


# size and power of the threshold test
def simulate_threshold_test(frame, shifts, reps, B_inner, rng, ctr=CONTROLS, gamma_q=0.6, year_share=0.0):
    G = grid(frame); n = len(frame)
    td = ThresholdDesign(frame, ctr, [np.full(n, g) for g in G])
    c, ng = td.c, td.ng
    Dv = frame.D.to_numpy(float)
    lin = fit(frame, "delta_E", ["D"] + ctr)
    XL, XLi = lin["X"], lin["XtXi"]
    sigma = float(np.sqrt(lin["rss"] / (n - lin["k"] - ng)))   # economy effects absorbed
    gtrue = float(np.quantile(Dv, gamma_q))
    yr = np.unique(frame.year.to_numpy(), return_inverse=True)[1]
    chi10 = stats.chi2.ppf(.90, 1)
    out = dict(design=dict(reps=reps, B_inner=B_inner, gamma_true=gtrue, gamma_quantile=gamma_q, year_share=year_share), rows=[])
    for shift in shifts:
        ps, naive10 = [], 0
        for _ in range(reps):
            y = rng.normal(0, 3, ng)[c] + shift * (Dv - gtrue) * (Dv > gtrue)
            if year_share:
                y = y + rng.normal(0, sigma * np.sqrt(year_share), yr.max() + 1)[yr] + rng.normal(0, sigma * np.sqrt(1 - year_share), n)
            else:
                y = y + rng.normal(0, sigma, n)
            Yd = demean(y[:, None], c, ng)
            i_hat, _ = td.search(Yd)
            w = td.shift_wald(Yd, i_hat)[0][0]
            naive10 += w > chi10
            bl = XLi @ (XL.T @ Yd[:, 0]); u0 = Yd[:, 0] - XL @ bl
            Ys = _null_draws(Yd[:, 0] - u0, u0, c, ng, B_inner, rng)
            idx, _ = td.search(Ys)
            ps.append((int(_at_least(td.shift_wald(Ys, idx)[0], w).sum()) + 1) / (B_inner + 1))
        ps = np.array(ps)
        out["rows"].append(dict(shift=float(shift), reject_10=float((ps <= .10).mean()), reject_05=float((ps <= .05).mean()),
                                textbook_reject_10=naive10 / reps))
    return out


# Driscoll-Kraay standard error
def driscoll_kraay(frame, y, xcols, term):
    r = fit(frame, y, xcols)
    h = r["X"] * r["u"][:, None]
    years = np.sort(frame.year.unique())
    Hs = np.vstack([h[frame.year.to_numpy() == yr].sum(0) for yr in years])
    T = len(years); m = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    S = Hs.T @ Hs
    for lag in range(1, m + 1):
        Gl = Hs[lag:].T @ Hs[:-lag]
        S = S + (1 - lag / (m + 1)) * (Gl + Gl.T)
    V = r["XtXi"] @ S @ r["XtXi"]
    j = r["xcols"].index(term)
    se = float(np.sqrt(V[j, j])); t = r["beta"][j] / se
    return dict(coef=float(r["beta"][j]), se=se, se_cluster=float(r["se"][term]), t=float(t), lags=m, T=T,
                p=float(2 * stats.norm.sf(abs(t))), p_upper=float(stats.norm.sf(t)))
