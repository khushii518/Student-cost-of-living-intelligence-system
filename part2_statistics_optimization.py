"""
STUDENT COST OF LIVING INTELLIGENCE SYSTEM
Part 2  |  Unit II (probability & statistics)  +  Unit IV (optimization, least squares, PCA)
Needs data/v2_features.csv from part 1.
"""
import os, json, warnings, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats, optimize
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
BASE = os.path.dirname(os.path.abspath(__file__))
F, R = os.path.join(BASE, "figures"), os.path.join(BASE, "results")
df = pd.read_csv(os.path.join(BASE, "data", "v2_features.csv"))
rng = np.random.default_rng(42)
OUT = {}          # numbers that go into the report
PVALS = []        # for multiple-testing correction


def addp(name, p):
    PVALS.append((name, float(p)))


# ====================================================================================
# UNIT II-A  PROBABILITY
# ====================================================================================
def probability_section():
    res = {}
    n = len(df)
    # empirical joint / marginal / conditional probabilities
    joint = pd.crosstab(df["Housing"], df["High_Stress"], normalize="all")
    marg_h = df["Housing"].value_counts(normalize=True)
    cond = pd.crosstab(df["Housing"], df["High_Stress"], normalize="index")[1]
    res["P(High stress)"] = round(df["High_Stress"].mean(), 3)
    res["P(Deficit)"] = round(df["Deficit"].mean(), 3)
    res["P(High stress | Housing)"] = cond.round(3).to_dict()
    cond_def = df.groupby("Housing")["Deficit"].mean().round(3).to_dict()
    res["P(Deficit | Housing)"] = cond_def
    # Bayes: P(PG | High stress) = P(HS|PG)P(PG)/P(HS)
    p_pg, p_hs = marg_h["PG"], df["High_Stress"].mean()
    p_hs_pg = cond["PG"]
    res["Bayes P(PG | High stress)"] = round(p_hs_pg * p_pg / p_hs, 3)
    res["P(High stress | Deficit=1)"] = round(df.loc[df.Deficit == 1, "High_Stress"].mean(), 3) if df.Deficit.sum() else None
    res["P(High stress | Deficit=0)"] = round(df.loc[df.Deficit == 0, "High_Stress"].mean(), 3)
    # independence tests (chi-square; Fisher for 2x2)
    ct = pd.crosstab(df["Housing"], df["High_Stress"])
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    res["chi2 Housing x HighStress"] = {"chi2": round(chi2, 3), "dof": int(dof), "p": round(p, 4),
                                       "min_expected_count": round(float(exp.min()), 2)}
    addp("Housing independent of High_Stress (chi2)", p)
    ct2 = pd.crosstab(df["Campus"], df["Deficit"])
    odds, p2 = stats.fisher_exact(ct2)
    res["Fisher Campus x Deficit"] = {"odds_ratio": round(float(odds), 3), "p": round(p2, 4)}
    addp("Campus independent of Deficit (Fisher)", p2)
    ct3 = pd.crosstab(df["Deficit"], df["High_Stress"])
    odds3, p3 = stats.fisher_exact(ct3)
    res["Fisher Deficit x HighStress"] = {"odds_ratio": round(float(odds3), 3), "p": round(p3, 4)}
    addp("Deficit independent of High_Stress (Fisher)", p3)
    # plot conditional probabilities
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    cond.reindex(["Home", "Hostel", "Flat", "PG"]).plot.bar(ax=ax[0], color="#e07a5f", rot=0)
    ax[0].set_title("P(High stress | housing)"); ax[0].set_ylim(0, 1)
    pd.Series(cond_def).reindex(["Home", "Hostel", "Flat", "PG"]).plot.bar(ax=ax[1], color="#3d405b", rot=0)
    ax[1].set_title("P(Spending > income | housing)"); ax[1].set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(os.path.join(F, "10_conditional_probability.png"), dpi=130); plt.close(fig)
    OUT["probability"] = res


# ====================================================================================
# UNIT II-B  RANDOM VARIABLES, DISTRIBUTIONS, EXPECTATION, MOMENTS
# ====================================================================================
def distributions_section():
    x = df["Computed_Total"].values.astype(float)
    res = {"mean(E[X])": x.mean(), "variance": x.var(ddof=1), "std": x.std(ddof=1),
           "skewness(3rd moment)": stats.skew(x), "excess kurtosis(4th)": stats.kurtosis(x),
           "median": np.median(x)}
    # fit candidate continuous distributions
    fits = {}
    for name, dist in {"normal": stats.norm, "lognormal": stats.lognorm, "gamma": stats.gamma,
                       "weibull": stats.weibull_min}.items():
        params = dist.fit(x, floc=0) if name != "normal" else dist.fit(x)
        ks = stats.kstest(x, dist.cdf, args=params)
        ll = dist.logpdf(x, *params).sum()
        k = len(params) - (1 if name != "normal" else 0)
        fits[name] = {"loglik": round(ll, 2), "AIC": round(2 * k - 2 * ll, 2),
                      "KS_p": round(ks.pvalue, 3), "params": [round(float(v), 3) for v in params]}
    res["fits"] = fits
    best = min(fits, key=lambda k: fits[k]["AIC"])
    res["best_by_AIC"] = best
    # non-gradient MLE (Nelder-Mead) of lognormal -> same answer as closed form
    nll = lambda th: -stats.lognorm.logpdf(x, s=abs(th[0]), scale=np.exp(th[1])).sum()
    nm = optimize.minimize(nll, [0.5, np.log(x.mean())], method="Nelder-Mead")
    res["NelderMead_lognormal"] = {"sigma": round(abs(nm.x[0]), 4), "mu": round(nm.x[1], 4),
                                   "closed_form_sigma": round(float(np.log(x).std()), 4),
                                   "closed_form_mu": round(float(np.log(x).mean()), 4)}
    sh = stats.shapiro(x)
    res["Shapiro-Wilk"] = {"W": round(sh.statistic, 3), "p": round(sh.pvalue, 4)}
    # discrete RV: number of paid-cost categories above 5k etc. Binomial / Poisson on Has_Side_Income
    p_side = df["Has_Side_Income"].mean()
    k = np.arange(0, 11)
    res["Binomial: P(>=3 of 10 random students have side income)"] = round(1 - stats.binom.cdf(2, 10, p_side), 3)
    res["Side-income prevalence p"] = round(p_side, 3)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.6))
    grid = np.linspace(x.min() * .6, x.max() * 1.1, 300)
    axs[0].hist(x, bins=14, density=True, alpha=.45, color="#81b29a")
    for name, dist in {"normal": stats.norm, "lognormal": stats.lognorm, "gamma": stats.gamma}.items():
        axs[0].plot(grid, dist.pdf(grid, *fits[name]["params"]), label=f"{name} (AIC {fits[name]['AIC']})")
    axs[0].set_title("Fitted probability density functions"); axs[0].legend()
    stats.probplot(x, dist="norm", plot=axs[1]); axs[1].set_title("Normal Q-Q plot")
    axs[2].plot(np.sort(x), np.arange(1, len(x) + 1) / len(x), drawstyle="steps-post", label="empirical CDF")
    axs[2].plot(grid, stats.lognorm.cdf(grid, *fits["lognormal"]["params"]), label="lognormal CDF")
    axs[2].set_title("Cumulative distribution function"); axs[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(F, "11_distribution_fitting.png"), dpi=130); plt.close(fig)
    OUT["distributions"] = res


# ====================================================================================
# UNIT II-C  RANDOM VECTORS, COVARIANCE, CORRELATION, MULTIVARIATE NORMAL
# ====================================================================================
def multivariate_section():
    cols = ["Rent", "Food", "Commute", "Academics"]
    X = df[cols].values.astype(float)
    mu, S = X.mean(0), np.cov(X.T)
    res = {"mean vector": dict(zip(cols, mu.round(1))),
           "covariance matrix": pd.DataFrame(S, index=cols, columns=cols).round(0).to_dict(),
           "correlation matrix": df[cols].corr().round(3).to_dict()}
    # conditional distribution Food | Rent under bivariate normal: E[F|R=r] = mu_F + S_FR/S_RR (r-mu_R)
    i, j = cols.index("Food"), cols.index("Rent")
    beta = S[i, j] / S[j, j]
    res["E[Food | Rent=r] slope"] = round(beta, 4)
    res["E[Food | Rent=14000]"] = round(mu[i] + beta * (14000 - mu[j]), 0)
    res["Var[Food | Rent] (conditional)"] = round(S[i, i] - S[i, j] ** 2 / S[j, j], 0)
    # Mahalanobis: multivariate normality / outliers
    Sinv = np.linalg.inv(S)
    d2 = np.einsum("ij,jk,ik->i", X - mu, Sinv, X - mu)
    df["Mahalanobis_d2"] = d2
    thr = stats.chi2.ppf(0.975, df=len(cols))
    res["Mahalanobis outliers (chi2 97.5%)"] = df.loc[d2 > thr, "ID"].tolist()
    # Correlation significance
    pairs = []
    for a, b in itertools.combinations(["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Stress", "Year"], 2):
        r, p = stats.pearsonr(df[a], df[b]); rs, ps = stats.spearmanr(df[a], df[b])
        pairs.append({"pair": f"{a} ~ {b}", "pearson_r": round(r, 3), "p": round(p, 4),
                      "spearman_rho": round(rs, 3), "p_spearman": round(ps, 4)})
        addp(f"corr {a}~{b}", p)
    pairs = pd.DataFrame(pairs).sort_values("p")
    pairs.to_csv(os.path.join(R, "correlation_tests.csv"), index=False)
    res["strongest correlations"] = pairs.head(6).to_dict("records")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))
    sorted_d = np.sort(d2); q = stats.chi2.ppf((np.arange(1, len(d2) + 1) - .5) / len(d2), df=len(cols))
    ax[0].scatter(q, sorted_d); ax[0].plot([0, q.max()], [0, q.max()], "r--")
    ax[0].set_xlabel("chi-square quantiles (df=4)"); ax[0].set_ylabel("ordered Mahalanobis d²")
    ax[0].set_title("Multivariate normality check")
    sns.regplot(data=df, x="Rent", y="Food", ax=ax[1], scatter_kws={"alpha": .6})
    ax[1].set_title(f"E[Food | Rent]  slope = {beta:.3f}")
    fig.tight_layout(); fig.savefig(os.path.join(F, "12_multivariate.png"), dpi=130); plt.close(fig)
    OUT["multivariate"] = res


# ====================================================================================
# UNIT II-D  SAMPLING DISTRIBUTIONS, CLT, ESTIMATION
# ====================================================================================
def clt_estimation_section():
    x = df["Computed_Total"].values.astype(float)
    res = {}
    fig, axs = plt.subplots(1, 4, figsize=(17, 3.8), sharey=False)
    axs[0].hist(x, bins=12, color="#e07a5f"); axs[0].set_title(f"Population (skew={stats.skew(x):.2f})")
    for a, n in zip(axs[1:], [2, 10, 30]):
        means = rng.choice(x, size=(5000, n), replace=True).mean(1)
        a.hist(means, bins=40, density=True, alpha=.6, color="#3d405b")
        g = np.linspace(means.min(), means.max(), 200)
        a.plot(g, stats.norm.pdf(g, x.mean(), x.std(ddof=1) / np.sqrt(n)), "r")
        a.set_title(f"Sample means, n={n} (skew={stats.skew(means):.2f})")
        res[f"CLT n={n}"] = {"mean_of_means": round(means.mean(), 0), "SE_empirical": round(means.std(), 0),
                             "SE_theory sigma/sqrt(n)": round(x.std(ddof=1) / np.sqrt(n), 0),
                             "skew": round(stats.skew(means), 3)}
    fig.suptitle("Central Limit Theorem demonstrated on the cost data"); fig.tight_layout()
    fig.savefig(os.path.join(F, "13_clt.png"), dpi=130); plt.close(fig)

    # sampling distribution of variance (normal population => (n-1)s²/σ² ~ chi2)
    n = len(x); s2 = x.var(ddof=1)
    lo, hi = (n - 1) * s2 / stats.chi2.ppf([.975, .025], n - 1)
    res["95% CI variance (chi2)"] = [round(lo, 0), round(hi, 0)]
    res["95% CI std (chi2)"] = [round(np.sqrt(lo), 0), round(np.sqrt(hi), 0)]
    # interval estimates of the mean: t, z, bootstrap
    se = x.std(ddof=1) / np.sqrt(n)
    res["mean"] = round(x.mean(), 0)
    res["95% CI mean (t)"] = [round(v, 0) for v in stats.t.interval(.95, n - 1, x.mean(), se)]
    res["95% CI mean (z)"] = [round(v, 0) for v in stats.norm.interval(.95, x.mean(), se)]
    boot = rng.choice(x, size=(10000, n)).mean(1)
    res["95% CI mean (bootstrap)"] = [round(v, 0) for v in np.percentile(boot, [2.5, 97.5])]
    # per-housing CI table
    rows = []
    for h, g in df.groupby("Housing"):
        v = g["Computed_Total"].values
        if len(v) >= 3:
            ci = stats.t.interval(.95, len(v) - 1, v.mean(), stats.sem(v))
        else:
            ci = (np.nan, np.nan)
        rows.append({"Housing": h, "n": len(v), "mean": round(v.mean()), "ci_low": round(ci[0]) if ci[0] == ci[0] else None,
                     "ci_high": round(ci[1]) if ci[1] == ci[1] else None})
    pd.DataFrame(rows).to_csv(os.path.join(R, "ci_by_housing.csv"), index=False)
    res["CI by housing"] = rows
    # proportion CI (Wilson) for deficit and high stress
    for col in ["Deficit", "High_Stress"]:
        k, n_ = int(df[col].sum()), len(df)
        ci = stats.binomtest(k, n_).proportion_ci(.95, method="wilson")
        res[f"95% Wilson CI P({col})"] = [round(ci.low, 3), round(ci.high, 3)]
    OUT["sampling_estimation"] = res


# ====================================================================================
# UNIT II-E  HYPOTHESIS TESTING + ANOVA (one-way, two-way)
# ====================================================================================
def cohen_d(a, b):
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp


def two_way_anova(y, A, B):
    """Type-II two-way ANOVA with interaction using nested OLS models (unbalanced-safe)."""
    def design(*cols):
        parts = [np.ones((len(y), 1))]
        for c in cols:
            parts.append(pd.get_dummies(c, drop_first=True).values.astype(float))
        return np.hstack(parts)
    def inter():
        a = pd.get_dummies(A, drop_first=True).values.astype(float)
        b = pd.get_dummies(B, drop_first=True).values.astype(float)
        return np.hstack([a[:, [i]] * b[:, [j]] for i in range(a.shape[1]) for j in range(b.shape[1])])
    def rss(X):
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ beta
        return float(r @ r), np.linalg.matrix_rank(X)
    XA, XB = design(A), design(B)
    XAB = design(A, B)
    XI = np.hstack([XAB, inter()])
    r0, k0 = rss(XAB); rf, kf = rss(XI); ra, ka = rss(XB); rb, kb = rss(XA)
    dfe = len(y) - kf
    mse = rf / dfe
    rows = []
    for name, ss, d in [("Campus (A)", ra - r0, k0 - ka), ("Housing (B)", rb - r0, k0 - kb), ("A x B interaction", r0 - rf, kf - k0)]:
        Fv = (ss / d) / mse if d > 0 else np.nan
        p = 1 - stats.f.cdf(Fv, d, dfe) if d > 0 else np.nan
        rows.append({"source": name, "SS": round(ss), "df": d, "MS": round(ss / d) if d else None,
                     "F": round(Fv, 3), "p": round(p, 4)})
    rows.append({"source": "Residual", "SS": round(rf), "df": dfe, "MS": round(mse), "F": None, "p": None})
    return pd.DataFrame(rows)


def hypothesis_section():
    res = {}
    y = df["Computed_Total"]
    # H1: two-sided Welch t-test Bidholi vs Kandoli
    a, b = y[df.Campus == "Bidholi"], y[df.Campus == "Kandoli"]
    t, p = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b)
    res["H1 Bidholi vs Kandoli cost (Welch t, two-tailed)"] = {
        "mean_B": round(a.mean()), "mean_K": round(b.mean()), "t": round(t, 3), "p": round(p, 4),
        "cohen_d": round(cohen_d(a, b), 3), "MannWhitney_p": round(u.pvalue, 4)}
    addp("H1 Welch t Campus cost", p)
    # H2: variance equality (F-test + Levene)
    Fv = a.var(ddof=1) / b.var(ddof=1)
    pF = 2 * min(stats.f.cdf(Fv, len(a) - 1, len(b) - 1), 1 - stats.f.cdf(Fv, len(a) - 1, len(b) - 1))
    lev = stats.levene(a, b)
    res["H2 equal variance Bidholi vs Kandoli"] = {"F": round(Fv, 3), "p_F": round(pF, 4), "Levene_p": round(lev.pvalue, 4)}
    # H3: ONE-TAILED - off-campus paying students have higher stress than hostel/home
    paying = df.loc[df.Is_Offcampus_Paying == 1, "Stress"]; not_pay = df.loc[df.Is_Offcampus_Paying == 0, "Stress"]
    t3, p3 = stats.ttest_ind(paying, not_pay, equal_var=False, alternative="greater")
    res["H3 one-tailed: paying-rent stress > hostel/home stress"] = {
        "mean_paying": round(paying.mean(), 2), "mean_other": round(not_pay.mean(), 2), "t": round(t3, 3), "p_one_tailed": round(p3, 4)}
    addp("H3 stress paying > not paying (1-tail)", p3)
    # H4: one-sample t-test vs a benchmark (INR 15,000 monthly benchmark)
    t4, p4 = stats.ttest_1samp(y, 15000)
    res["H4 one-sample: mean cost vs INR 15,000 benchmark"] = {"t": round(t4, 3), "p": round(p4, 4), "mean": round(y.mean())}
    addp("H4 mean cost vs 15000", p4)
    # H5: Z-test for a proportion  H0: P(deficit)=0.30
    k, n = int(df.Deficit.sum()), len(df); p0 = 0.30
    z = (k / n - p0) / np.sqrt(p0 * (1 - p0) / n); p5 = 2 * (1 - stats.norm.cdf(abs(z)))
    res["H5 Z-test: P(deficit)=0.30"] = {"observed": round(k / n, 3), "z": round(z, 3), "p": round(p5, 4)}
    addp("H5 z-test deficit prop", p5)
    # H6: paired t-test: Reported vs Computed total
    t6, p6 = stats.ttest_rel(df.Reported_Total, df.Computed_Total)
    res["H6 paired: reported vs itemised total"] = {"mean_diff": round((df.Reported_Total - df.Computed_Total).mean()), "t": round(t6, 3), "p": round(p6, 4)}
    addp("H6 paired reported vs computed", p6)
    # One-way ANOVA across streams + housing; Kruskal; Tukey
    for g in ["Housing", "Stream_Group", "Year"]:
        groups = [v["Computed_Total"].values for _, v in df.groupby(g) if len(v) > 1]
        Fv, pv = stats.f_oneway(*groups)
        kw = stats.kruskal(*groups)
        ss_b = sum(len(x) * (x.mean() - y.mean()) ** 2 for x in groups); ss_t = ((y - y.mean()) ** 2).sum()
        res[f"One-way ANOVA cost ~ {g}"] = {"F": round(Fv, 3), "p": round(pv, 4), "eta_sq": round(ss_b / ss_t, 3), "Kruskal_p": round(kw.pvalue, 4)}
        addp(f"ANOVA cost ~ {g}", pv)
    g4 = [v["Computed_Total"].values for k, v in df.groupby("Housing") if len(v) > 1]
    names = [k for k, v in df.groupby("Housing") if len(v) > 1]
    tk = stats.tukey_hsd(*g4)
    tuk = []
    for i, j in itertools.combinations(range(len(names)), 2):
        tuk.append({"pair": f"{names[i]} vs {names[j]}", "mean_diff": round(float(g4[i].mean() - g4[j].mean())),
                    "p_adj": round(float(tk.pvalue[i, j]), 4)})
    pd.DataFrame(tuk).to_csv(os.path.join(R, "tukey_housing.csv"), index=False)
    res["Tukey HSD housing"] = tuk
    # ANOVA on stress by housing
    gs = [v["Stress"].values for _, v in df.groupby("Housing") if len(v) > 1]
    Fs, ps = stats.f_oneway(*gs)
    res["One-way ANOVA stress ~ Housing"] = {"F": round(Fs, 3), "p": round(ps, 4)}
    addp("ANOVA stress ~ Housing", ps)
    # Two-way ANOVA
    df["Housing3"] = df["Housing"].replace({"Home": "Hostel/Home"}).replace({"Hostel": "Hostel/Home"})
    tw = two_way_anova(y.values.astype(float), df["Campus"].values, df["Housing"].values)
    tw.to_csv(os.path.join(R, "two_way_anova.csv"), index=False)
    res["Two-way ANOVA cost ~ Campus x Housing"] = tw.to_dict("records")
    for r_ in tw.dropna(subset=["p"]).to_dict("records"):
        addp(f"two-way ANOVA {r_['source']}", r_["p"])
    # Multiple-testing correction (Holm)
    names_p = [n for n, _ in PVALS]; pv = np.array([p for _, p in PVALS])
    order = np.argsort(pv); m = len(pv); holm = np.empty(m); run = 0
    for rank, idx in enumerate(order):
        run = max(run, min(1, (m - rank) * pv[idx])); holm[idx] = run
    hdf = pd.DataFrame({"test": names_p, "p_raw": pv.round(4), "p_holm": holm.round(4), "significant_5pct_after_holm": holm < .05}).sort_values("p_raw")
    hdf.to_csv(os.path.join(R, "all_tests_holm.csv"), index=False)
    res["tests_run"] = m; res["significant_after_holm"] = hdf[hdf.significant_5pct_after_holm].test.tolist()

    # plot
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    sns.boxplot(data=df, x="Housing", y="Computed_Total", order=["Home", "Hostel", "Flat", "PG"], ax=ax[0], palette="Set2")
    sns.stripplot(data=df, x="Housing", y="Computed_Total", order=["Home", "Hostel", "Flat", "PG"], ax=ax[0], color="k", alpha=.5)
    ax[0].set_title("One-way ANOVA: cost by housing")
    sns.pointplot(data=df, x="Housing", y="Computed_Total", hue="Campus", order=["Home", "Hostel", "Flat", "PG"], ax=ax[1], dodge=.3, errorbar="se")
    ax[1].set_title("Two-way ANOVA interaction plot (mean ± SE)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "14_anova.png"), dpi=130); plt.close(fig)
    OUT["hypothesis"] = res


# ====================================================================================
# UNIT IV  OPTIMIZATION: least squares, gradient descent, PCA, constrained budget
# ====================================================================================
def gradient_descent_section():
    res = {}
    feats = ["Allowance", "SideIncome", "Year", "Has_Side_Income", "Is_Offcampus_Paying"]
    X = df[feats].values.astype(float); y = df["Computed_Total"].values.astype(float)
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    Xb = np.hstack([np.ones((len(y), 1)), Xs])
    ym, ys = y.mean(), y.std(); yt = (y - ym) / ys
    # closed form least squares (normal equation) and QR/SVD lstsq
    beta_ne = np.linalg.solve(Xb.T @ Xb, Xb.T @ yt)
    beta_ls, *_ = np.linalg.lstsq(Xb, yt, rcond=None)
    # batch gradient descent
    def gd(lr, mom=0.0, epochs=400):
        b = np.zeros(Xb.shape[1]); v = np.zeros_like(b); hist = []
        for _ in range(epochs):
            g = (2 / len(yt)) * Xb.T @ (Xb @ b - yt)
            v = mom * v - lr * g; b = b + v
            hist.append(float(np.mean((Xb @ b - yt) ** 2)))
        return b, hist
    b1, h1 = gd(0.05); b2, h2 = gd(0.05, 0.9); b3, h3 = gd(0.3)
    # stochastic (mini-batch) GD
    b = np.zeros(Xb.shape[1]); hs = []
    for ep in range(200):
        idx = rng.permutation(len(yt))
        for s in range(0, len(yt), 10):
            bi = idx[s:s + 10]
            b -= 0.02 * (2 / len(bi)) * Xb[bi].T @ (Xb[bi] @ b - yt[bi])
        hs.append(float(np.mean((Xb @ b - yt) ** 2)))
    sk = LinearRegression().fit(Xs, yt)
    res["features"] = feats
    res["coef normal-equation"] = beta_ne.round(4).tolist()
    res["coef GD(lr=.05)"] = b1.round(4).tolist(); res["coef GD+momentum"] = b2.round(4).tolist()
    res["coef mini-batch SGD"] = b.round(4).tolist()
    res["coef sklearn"] = np.r_[sk.intercept_, sk.coef_].round(4).tolist()
    res["max |GD - normal eq|"] = round(float(np.abs(b1 - beta_ne).max()), 5)
    res["final MSE (standardised) normal-eq"] = round(float(np.mean((Xb @ beta_ne - yt) ** 2)), 4)
    res["R2 in-sample"] = round(1 - np.mean((Xb @ beta_ne - yt) ** 2) / np.var(yt), 3)
    res["Condition number of design matrix"] = round(float(np.linalg.cond(Xb)), 2)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(h1, label="GD lr=0.05"); ax.plot(h2, label="GD + momentum 0.9"); ax.plot(h3, label="GD lr=0.3")
    ax.plot(np.linspace(0, 400, len(hs)), hs, label="mini-batch SGD", alpha=.7)
    ax.axhline(np.mean((Xb @ beta_ne - yt) ** 2), color="k", ls="--", label="least-squares optimum")
    ax.set_yscale("log"); ax.set_xlabel("iteration / epoch"); ax.set_ylabel("MSE (log)"); ax.legend(); ax.set_title("Gradient-descent convergence vs closed-form least squares")
    fig.tight_layout(); fig.savefig(os.path.join(F, "15_gradient_descent.png"), dpi=130); plt.close(fig)
    OUT["gradient_descent"] = res


def feature_selection_section():
    res = {}
    cand = ["Allowance", "SideIncome", "Year", "Age", "Has_Side_Income", "Is_Offcampus_Paying", "Total_Income",
            "Self_Funded_Share", "Stress"]
    Xc = df[cand].astype(float)
    y = df["Computed_Total"].astype(float)
    mi = mutual_info_regression(StandardScaler().fit_transform(Xc), y, random_state=0)
    cor = Xc.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]))
    sel = pd.DataFrame({"mutual_info": mi.round(3), "abs_corr": cor.round(3)}, index=cand).sort_values("mutual_info", ascending=False)
    # forward selection by adjusted R2 (subset selection)
    chosen, remaining, best = [], cand.copy(), -1
    trace = []
    n = len(y)
    while remaining:
        scores = {}
        for c in remaining:
            Xm = np.hstack([np.ones((n, 1)), Xc[chosen + [c]].values]); b, *_ = np.linalg.lstsq(Xm, y, rcond=None)
            r2 = 1 - ((y - Xm @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum(); k = Xm.shape[1] - 1
            scores[c] = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        c, s = max(scores.items(), key=lambda t: t[1])
        if s <= best + 1e-4: break
        chosen.append(c); remaining.remove(c); best = s; trace.append({"added": c, "adj_R2": round(s, 3)})
    res["filter ranking (MI / corr)"] = sel.reset_index().rename(columns={"index": "feature"}).to_dict("records")
    res["forward selection"] = trace
    sel.to_csv(os.path.join(R, "feature_selection_filter.csv"))
    # PCA - feature reduction on the six money variables + stress
    pc_cols = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Stress"]
    Z = StandardScaler().fit_transform(df[pc_cols]); pca = PCA().fit(Z)
    ev = pca.explained_variance_ratio_
    res["PCA explained variance"] = ev.round(3).tolist(); res["PCs for 80%"] = int(np.searchsorted(np.cumsum(ev), .80) + 1)
    load = pd.DataFrame(pca.components_[:3].T, index=pc_cols, columns=["PC1", "PC2", "PC3"]).round(3)
    load.to_csv(os.path.join(R, "pca_loadings.csv")); res["PCA loadings"] = load.to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].bar(range(1, len(ev) + 1), ev, color="#81b29a"); ax[0].plot(range(1, len(ev) + 1), np.cumsum(ev), "ko-")
    ax[0].set_xlabel("component"); ax[0].set_title("PCA scree + cumulative variance")
    sns.heatmap(load, annot=True, cmap="RdBu_r", center=0, ax=ax[1]); ax[1].set_title("PCA loadings")
    fig.tight_layout(); fig.savefig(os.path.join(F, "16_pca_scree_loadings.png"), dpi=130); plt.close(fig)
    OUT["feature_selection"] = res


def constrained_optimisation_section():
    """Prescriptive analytics: best budget split given a monthly budget.
    maximise  sum_i w_i * log(x_i - floor_i)   s.t.  sum x_i <= B,  x_i >= floor_i
    floors = 10th percentile of observed spend in that housing type; w = observed spend shares."""
    cats = ["Rent", "Food", "Commute", "Academics"]
    rows = []
    for housing in ["Hostel", "PG", "Flat"]:
        g = df[df.Housing == housing]
        floors = g[cats].quantile(.10).values
        floors = np.maximum(floors, [0, 2500, 500, 1000])      # sensible survival floors
        w = g[cats].mean().values / g[cats].mean().values.sum()
        if housing == "Hostel":
            w = np.array([1e-9, *w[1:]]); floors[0] = 0        # hostel fee not in the survey
        for B in [10000, 15000, 20000, 30000]:
            if B <= floors.sum() + 500:
                continue
            obj = lambda x: -np.sum(w[1:] * np.log(np.maximum(x[1:] - floors[1:], 1e-6))) - (w[0] * np.log(max(x[0] - floors[0], 1e-6)) if w[0] > 1e-6 else 0)
            cons = [{"type": "ineq", "fun": lambda x, B=B: B - x.sum()}]
            bnds = [(floors[i], None) for i in range(4)]
            x0 = floors + (B - floors.sum()) * w
            sol = optimize.minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP")
            x = sol.x
            rows.append({"Housing": housing, "Budget": B, **{c: round(v / 50) * 50 for c, v in zip(cats, x)},
                         "converged": bool(sol.success)})
    plan = pd.DataFrame(rows)
    plan.to_csv(os.path.join(R, "optimal_budget_plan.csv"), index=False)
    OUT["constrained_optimisation"] = plan.to_dict("records")
    # Nelder-Mead / Powell (non-gradient) check on one case
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, h in zip(axs, ["Hostel", "PG", "Flat"]):
        p = plan[plan.Housing == h].set_index("Budget")[cats]
        if len(p): p.plot.bar(stacked=True, ax=ax, colormap="Set2", rot=0, legend=(h == "Hostel"))
        ax.set_title(f"Optimal split - {h}")
    axs[0].set_ylabel("INR / month")
    fig.tight_layout(); fig.savefig(os.path.join(F, "17_optimal_budget.png"), dpi=130); plt.close(fig)


# ====================================================================================
if __name__ == "__main__":
    probability_section(); distributions_section(); multivariate_section()
    clt_estimation_section(); hypothesis_section()
    gradient_descent_section(); feature_selection_section(); constrained_optimisation_section()
    json.dump(OUT, open(os.path.join(R, "part2_results.json"), "w"), indent=2, default=str)
    print(json.dumps({k: OUT[k] for k in ["probability", "hypothesis"]}, indent=1, default=str)[:6000])
