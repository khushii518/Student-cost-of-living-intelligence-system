"""
STUDENT COST OF LIVING INTELLIGENCE SYSTEM
Part 3  |  Unit V (supervised, unsupervised, association rules, anomaly detection, evaluation)
Needs data/v2_features.csv from part 1.
"""
import os, json, warnings, itertools
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, cross_validate, cross_val_predict, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, LocalOutlierFactor
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (confusion_matrix, roc_curve, roc_auc_score, silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score, adjusted_rand_score, silhouette_samples, ConfusionMatrixDisplay)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial.distance import pdist

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
BASE = os.path.dirname(os.path.abspath(__file__))
F, R = os.path.join(BASE, "figures"), os.path.join(BASE, "results")
df = pd.read_csv(os.path.join(BASE, "data", "v2_features.csv"))
OUT = {}
SEED = 42


# ==================================================================================
# SUPERVISED 1 - REGRESSION  (predict monthly itemised cost from context only)
# ==================================================================================
def regression_models():
    num = ["Allowance", "SideIncome", "Year", "Age"]
    cat = ["Campus", "Housing", "Stream_Group", "Gender"]
    X = df[num + cat]; y = df["Computed_Total"].astype(float)
    pre = ColumnTransformer([("n", StandardScaler(), num),
                             ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    models = {
        "Baseline (mean)": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge (a=3)": Ridge(alpha=3),
        "Decision Tree (d=3)": DecisionTreeRegressor(max_depth=3, min_samples_leaf=3, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, random_state=SEED),
        "SVR (RBF)": SVR(C=30000, epsilon=1000, gamma="scale"),
        "kNN (k=5)": KNeighborsRegressor(5, weights="distance"),
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows = []
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("m", m)])
        s = cross_validate(pipe, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"))
        rows.append({"model": name, "R2": s["test_r2"].mean(), "R2_sd": s["test_r2"].std(),
                     "MAE_INR": -s["test_neg_mean_absolute_error"].mean(), "RMSE_INR": -s["test_neg_root_mean_squared_error"].mean()})
    t = pd.DataFrame(rows).round(3).sort_values("R2", ascending=False)
    t.to_csv(os.path.join(R, "regression_cv.csv"), index=False)
    best = t.iloc[0]["model"]
    # permutation importance with RF on whole data (for interpretation, not for scoring)
    rf = Pipeline([("pre", pre), ("m", models["Random Forest"])]).fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=SEED)
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    # linear coefficients (readable)
    lin = Pipeline([("pre", pre), ("m", LinearRegression())]).fit(X, y)
    names = lin["pre"].get_feature_names_out()
    coefs = pd.Series(lin["m"].coef_, index=names).sort_values()
    coefs.to_csv(os.path.join(R, "linear_regression_coefficients.csv"))
    pred = cross_val_predict(Pipeline([("pre", pre), ("m", Ridge(alpha=3))]), X, y, cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=SEED))
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    t.set_index("model")["R2"].sort_values().plot.barh(ax=ax[0], color="#81b29a"); ax[0].axvline(0, color="k")
    ax[0].set_title("Cost regression: cross-validated R² (5-fold x10)")
    imp.plot.barh(ax=ax[1], color="#e07a5f"); ax[1].set_title("Permutation importance (Random Forest)")
    ax[2].scatter(y, pred, alpha=.7); lim = [0, y.max()]; ax[2].plot(lim, lim, "k--")
    ax[2].set_xlabel("actual"); ax[2].set_ylabel("cross-validated prediction (Ridge)"); ax[2].set_title("Actual vs predicted")
    fig.tight_layout(); fig.savefig(os.path.join(F, "18_regression.png"), dpi=130); plt.close(fig)
    # decision-tree picture
    tree = Pipeline([("pre", pre), ("m", DecisionTreeRegressor(max_depth=3, min_samples_leaf=4, random_state=SEED))]).fit(X, y)
    fig, ax = plt.subplots(figsize=(17, 6.5))
    plot_tree(tree["m"], feature_names=tree["pre"].get_feature_names_out(), filled=True, rounded=True, fontsize=8, ax=ax, precision=0)
    ax.set_title("Decision tree: what drives monthly cost?")
    fig.tight_layout(); fig.savefig(os.path.join(F, "19_decision_tree_cost.png"), dpi=130); plt.close(fig)
    OUT["regression"] = {"cv_table": t.to_dict("records"), "best": best,
                         "perm_importance": imp.sort_values(ascending=False).round(3).to_dict(),
                         "top_coefficients": {"most_negative": coefs.head(3).round(0).to_dict(), "most_positive": coefs.tail(3).round(0).to_dict()}}


# ==================================================================================
# SUPERVISED 2 - CLASSIFICATION (predict high financial stress)
# ==================================================================================
def classification_models():
    num = ["Allowance", "SideIncome", "Year", "Spend_to_Income", "Savings_Rate", "Rent_Share", "Food_Share", "Computed_Total"]
    cat = ["Housing", "Campus"]
    X = df[num + cat]; y = df["High_Stress"]
    pre = ColumnTransformer([("n", StandardScaler(), num),
                             ("c", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])
    models = {
        "Baseline (majority)": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": LogisticRegression(C=0.5, max_iter=2000),
        "Decision Tree (d=3)": DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=SEED),
        "SVM (RBF)": SVC(C=1, probability=True, class_weight="balanced", random_state=SEED),
        "k-NN (k=7)": KNeighborsClassifier(7),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    rows = []
    for name, m in models.items():
        pipe = Pipeline([("pre", pre), ("m", m)])
        s = cross_validate(pipe, X, y, cv=cv, scoring=("accuracy", "f1", "roc_auc", "recall", "precision"))
        rows.append({"model": name, "accuracy": s["test_accuracy"].mean(), "precision": s["test_precision"].mean(),
                     "recall": s["test_recall"].mean(), "F1": s["test_f1"].mean(), "ROC_AUC": s["test_roc_auc"].mean(),
                     "AUC_sd": s["test_roc_auc"].std()})
    t = pd.DataFrame(rows).round(3).sort_values("ROC_AUC", ascending=False)
    t.to_csv(os.path.join(R, "classification_cv.csv"), index=False)

    # ROC curves + confusion matrix (cross-validated predictions)
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
    cv1 = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED)
    for name in ["Logistic Regression", "Random Forest", "SVM (RBF)", "k-NN (k=7)", "Decision Tree (d=3)"]:
        pr = cross_val_predict(Pipeline([("pre", pre), ("m", models[name])]), X, y, cv=cv1, method="predict_proba")[:, 1]
        fpr, tpr, _ = roc_curve(y, pr); ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y, pr):.2f})")
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].legend(fontsize=8); ax[0].set_title("ROC curves (cross-validated)")
    best_name = t.iloc[0]["model"]; best_name = best_name if best_name != "Baseline (majority)" else t.iloc[1]["model"]
    pred = cross_val_predict(Pipeline([("pre", pre), ("m", models[best_name])]), X, y, cv=cv1)
    ConfusionMatrixDisplay(confusion_matrix(y, pred), display_labels=["OK", "High stress"]).plot(ax=ax[1], colorbar=False, cmap="Blues")
    ax[1].set_title(f"Confusion matrix - {best_name}")
    rf = Pipeline([("pre", pre), ("m", models["Random Forest"])]).fit(X, y)
    pi = permutation_importance(rf, X, y, n_repeats=30, random_state=SEED, scoring="roc_auc")
    imp = pd.Series(pi.importances_mean, index=X.columns).sort_values()
    imp.plot.barh(ax=ax[2], color="#3d405b"); ax[2].set_title("Permutation importance (stress classifier)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "20_classification.png"), dpi=130); plt.close(fig)

    # simple hyper-parameter tuning curve for k-NN and tree depth (CV)
    ks = list(range(1, 16, 2))
    gs = GridSearchCV(Pipeline([("pre", pre), ("m", KNeighborsClassifier())]), {"m__n_neighbors": ks}, cv=cv, scoring="roc_auc").fit(X, y)
    OUT["classification"] = {"cv_table": t.to_dict("records"), "best_model": best_name,
                             "confusion_matrix": confusion_matrix(y, pred).tolist(),
                             "perm_importance_auc": imp.sort_values(ascending=False).round(3).to_dict(),
                             "knn_best_k": int(gs.best_params_["m__n_neighbors"]), "knn_best_auc": round(gs.best_score_, 3),
                             "class_balance": y.value_counts().to_dict()}


# ==================================================================================
# UNSUPERVISED - K-Means, Hierarchical, DBSCAN, GMM, PCA, t-SNE
# ==================================================================================
def clustering_section():
    cols = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Stress", "Savings_Rate"]
    Xs = StandardScaler().fit_transform(df[cols]); res = {}
    ks = range(2, 9); inertia, sil, dbi, chi = [], [], [], []
    for k in ks:
        km = KMeans(k, n_init=20, random_state=SEED).fit(Xs); lab = km.labels_
        inertia.append(km.inertia_); sil.append(silhouette_score(Xs, lab)); dbi.append(davies_bouldin_score(Xs, lab)); chi.append(calinski_harabasz_score(Xs, lab))
    tab = pd.DataFrame({"k": list(ks), "inertia": inertia, "silhouette": sil, "davies_bouldin": dbi, "calinski_harabasz": chi}).round(3)
    tab.to_csv(os.path.join(R, "kmeans_model_selection.csv"), index=False)
    # choose best silhouette among k in 3..5 (k=2 is usually a trivial split)
    # rule: k in 3..5, every cluster must hold >= 4 students (avoid singleton "clusters"), then best silhouette
    ok = [k for k in range(3, 6) if np.bincount(KMeans(k, n_init=50, random_state=SEED).fit(Xs).labels_).min() >= 4]
    sub = tab[tab.k.isin(ok)]; K = int(sub.loc[sub.silhouette.idxmax(), "k"])
    km = KMeans(K, n_init=50, random_state=SEED).fit(Xs); df["Cluster_KMeans"] = km.labels_
    res["K chosen"] = K; res["model_selection"] = tab.to_dict("records")
    res["silhouette"] = round(silhouette_score(Xs, km.labels_), 3)
    # hierarchical
    Z = linkage(Xs, "ward"); coph = cophenet(Z, pdist(Xs))[0]
    hc = AgglomerativeClustering(K, linkage="ward").fit(Xs); df["Cluster_Hier"] = hc.labels_
    res["cophenetic corr (ward)"] = round(float(coph), 3); res["ARI KMeans vs Hierarchical"] = round(adjusted_rand_score(km.labels_, hc.labels_), 3)
    # DBSCAN with k-distance guide
    kd = np.sort(NearestNeighbors(n_neighbors=4).fit(Xs).kneighbors(Xs)[0][:, -1])
    eps = float(np.percentile(kd, 85))
    db = DBSCAN(eps=eps, min_samples=4).fit(Xs); df["Cluster_DBSCAN"] = db.labels_
    res["DBSCAN eps"] = round(eps, 2); res["DBSCAN clusters"] = int(len(set(db.labels_)) - (1 if -1 in db.labels_ else 0))
    res["DBSCAN noise points"] = int((db.labels_ == -1).sum())
    # Gaussian Mixture - BIC for K
    bic = {k: GaussianMixture(k, covariance_type="diag", n_init=10, random_state=SEED).fit(Xs).bic(Xs) for k in range(1, 6)}
    kg = min(bic, key=bic.get); gmm = GaussianMixture(K, covariance_type="diag", n_init=10, random_state=SEED).fit(Xs)
    df["Cluster_GMM"] = gmm.predict(Xs); df["GMM_MaxProb"] = gmm.predict_proba(Xs).max(1)
    res["GMM best K by BIC (K<=5, n=50)"] = kg; res["BIC"] = {k: round(v, 1) for k, v in bic.items()}
    res["ARI KMeans vs GMM"] = round(adjusted_rand_score(km.labels_, df["Cluster_GMM"]), 3)
    res["GMM uncertain students (max prob<0.7)"] = int((df["GMM_MaxProb"] < .7).sum())

    # ---- persona profiles
    prof = df.groupby("Cluster_KMeans").agg(n=("ID", "count"), Allowance=("Allowance", "mean"), SideIncome=("SideIncome", "mean"),
                                            Rent=("Rent", "mean"), Food=("Food", "mean"), Commute=("Commute", "mean"),
                                            Academics=("Academics", "mean"), Cost=("Computed_Total", "mean"),
                                            Stress=("Stress", "mean"), Savings_Rate=("Savings_Rate", "mean"),
                                            Hostel=("Housing", lambda s: (s == "Hostel").mean()), PG=("Housing", lambda s: (s == "PG").mean()),
                                            Flat=("Housing", lambda s: (s == "Flat").mean()), Home=("Housing", lambda s: (s == "Home").mean()),
                                            Bidholi=("Campus", lambda s: (s == "Bidholi").mean())).round(2)
    prof.to_csv(os.path.join(R, "cluster_profiles.csv")); res["profiles"] = prof.reset_index().to_dict("records")

    # ---- plots
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(tab.k, tab.inertia, "o-"); ax[0].set_title("Elbow (inertia)"); ax[0].set_xlabel("k")
    ax[1].plot(tab.k, tab.silhouette, "o-", color="#e07a5f"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k")
    ax[2].plot(list(bic), list(bic.values()), "o-", color="#3d405b"); ax[2].set_title("GMM BIC"); ax[2].set_xlabel("components")
    fig.tight_layout(); fig.savefig(os.path.join(F, "21_cluster_selection.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    dendrogram(Z, ax=ax, color_threshold=Z[-(K - 1), 2], labels=df["ID"].values, leaf_font_size=7)
    ax.set_title(f"Hierarchical clustering (Ward) - cut at {K} clusters")
    fig.tight_layout(); fig.savefig(os.path.join(F, "22_dendrogram.png"), dpi=130); plt.close(fig)

    pca = PCA(2, random_state=SEED).fit(Xs); P = pca.transform(Xs)
    tsne = TSNE(2, perplexity=10, init="pca", learning_rate="auto", random_state=SEED).fit_transform(Xs)
    df["PC1"], df["PC2"], df["tSNE1"], df["tSNE2"] = P[:, 0], P[:, 1], tsne[:, 0], tsne[:, 1]
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    for a, (x1, x2, ttl) in zip(ax[0], [("PC1", "PC2", f"PCA ({pca.explained_variance_ratio_.sum():.0%} variance)"), ("tSNE1", "tSNE2", "t-SNE (perplexity 10)")]):
        sns.scatterplot(data=df, x=x1, y=x2, hue="Cluster_KMeans", palette="Set1", s=90, ax=a, edgecolor="k"); a.set_title(f"{ttl} - K-Means clusters")
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster_DBSCAN", palette="tab10", s=90, ax=ax[1][0], edgecolor="k"); ax[1][0].set_title("PCA - DBSCAN (-1 = noise)")
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Cluster_GMM", palette="Set2", size="GMM_MaxProb", ax=ax[1][1], edgecolor="k"); ax[1][1].set_title("PCA - Gaussian Mixture (size = confidence)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "23_clusters_pca_tsne.png"), dpi=130); plt.close(fig)

    # radar-ish heatmap of cluster profile (z-scored means)
    hm = prof[["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Stress", "Savings_Rate"]]
    hm = (hm - hm.mean()) / hm.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(9, 3.5 + .4 * K)); sns.heatmap(hm, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Cluster personas (z-scored cluster means)"); fig.tight_layout()
    fig.savefig(os.path.join(F, "24_cluster_personas.png"), dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); sv = silhouette_samples(Xs, km.labels_)
    y0 = 0
    for c in range(K):
        v = np.sort(sv[km.labels_ == c]); ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, alpha=.7); ax.text(-.08, y0 + len(v) / 2, str(c)); y0 += len(v) + 2
    ax.axvline(sv.mean(), color="r", ls="--"); ax.set_title(f"Silhouette plot (mean {sv.mean():.2f})")
    fig.tight_layout(); fig.savefig(os.path.join(F, "25_silhouette_plot.png"), dpi=130); plt.close(fig)
    OUT["clustering"] = res


# ==================================================================================
# ASSOCIATION RULES - Apriori implemented from scratch
# ==================================================================================
def apriori(transactions, min_support=0.12, max_len=3):
    n = len(transactions); items = sorted({i for t in transactions for i in t})
    support = lambda s: sum(1 for t in transactions if s <= t) / n
    freq = {}; cur = [frozenset([i]) for i in items if support(frozenset([i])) >= min_support]
    for c in cur: freq[c] = support(c)
    k = 2
    while cur and k <= max_len:
        cand = {a | b for a in cur for b in cur if len(a | b) == k}
        cur = [c for c in cand if support(c) >= min_support]
        for c in cur: freq[c] = support(c)
        k += 1
    return freq


def association_rules_section():
    t = pd.DataFrame(index=df.index)
    t["Housing=" + df["Housing"]] = 0
    items = []
    for i, r in df.iterrows():
        s = {f"Housing={r.Housing}", f"Campus={r.Campus}",
             f"Cost={'High' if r.Computed_Total >= df.Computed_Total.quantile(.67) else 'Low' if r.Computed_Total <= df.Computed_Total.quantile(.33) else 'Mid'}",
             f"Stress={'High' if r.Stress >= 7 else 'Low' if r.Stress <= 3 else 'Mid'}",
             f"Income={'High' if r.Total_Income >= df.Total_Income.quantile(.67) else 'Low' if r.Total_Income <= df.Total_Income.quantile(.33) else 'Mid'}",
             f"Side={'Yes' if r.Has_Side_Income else 'No'}",
             f"Balance={'Deficit' if r.Deficit else 'Surplus'}",
             f"Food={'High' if r.Food_Share >= df.Food_Share.quantile(.67) else 'NotHigh'}"}
        items.append(frozenset(s))
    freq = apriori(items, min_support=0.12, max_len=3)
    rules = []
    for itemset, sup in freq.items():
        if len(itemset) < 2: continue
        for r in range(1, len(itemset)):
            for ante in map(frozenset, itertools.combinations(itemset, r)):
                cons = itemset - ante; conf = sup / freq[ante]; lift = conf / freq[cons]
                rules.append({"antecedent": " & ".join(sorted(ante)), "consequent": " & ".join(sorted(cons)),
                              "support": sup, "confidence": conf, "lift": lift, "leverage": sup - freq[ante] * freq[cons],
                              "conviction": (1 - freq[cons]) / (1 - conf) if conf < 1 else np.inf})
    rd = pd.DataFrame(rules)
    rd = rd[(rd.confidence >= .7) & (rd.lift > 1.2)].sort_values(["lift", "confidence"], ascending=False)
    rd = rd[rd.consequent.str.startswith(("Stress", "Balance", "Cost"))].round(3).head(60)
    rd.to_csv(os.path.join(R, "association_rules.csv"), index=False)
    focus = rd[rd.consequent.str.contains("Stress=High|Balance=Deficit")].head(8)
    OUT["association"] = {"frequent_itemsets": len(freq), "rules_kept": len(rd), "top_rules": rd.head(10).to_dict("records"),
                          "stress_or_deficit_rules": focus.to_dict("records")}
    top = rd.head(12)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sc = ax.scatter(rd.support, rd.confidence, c=rd.lift, s=60 + 200 * rd.support, cmap="viridis", edgecolor="k")
    plt.colorbar(sc, label="lift"); ax.set_xlabel("support"); ax.set_ylabel("confidence"); ax.set_title("Association rules (Apriori, from scratch)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "26_association_rules.png"), dpi=130); plt.close(fig)


# ==================================================================================
# ANOMALY DETECTION - Isolation Forest, LOF, Mahalanobis, robust z
# ==================================================================================
def anomaly_section():
    cols = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Computed_Total", "Stress", "Savings_Rate"]
    Xs = StandardScaler().fit_transform(df[cols])
    iso = IsolationForest(n_estimators=500, contamination=0.1, random_state=SEED).fit(Xs)
    df["Anom_IsoForest"] = (iso.predict(Xs) == -1).astype(int); df["Anom_Score"] = -iso.score_samples(Xs)
    lof = LocalOutlierFactor(n_neighbors=8, contamination=0.1); df["Anom_LOF"] = (lof.fit_predict(Xs) == -1).astype(int)
    Xm = df[cols[:6]].values; S = np.cov(Xm.T); d2 = np.einsum("ij,jk,ik->i", Xm - Xm.mean(0), np.linalg.pinv(S), Xm - Xm.mean(0))
    from scipy.stats import chi2
    df["Anom_Mahal"] = (d2 > chi2.ppf(.975, len(cols[:6]))).astype(int)
    med = df[cols].median(); mad = (df[cols] - med).abs().median().replace(0, 1)
    rz = ((df[cols] - med) / (1.4826 * mad)).abs()
    df["Anom_RobustZ"] = (rz.max(axis=1) > 3.5).astype(int)
    df["Anom_Votes"] = df[["Anom_IsoForest", "Anom_LOF", "Anom_Mahal", "Anom_RobustZ"]].sum(axis=1)
    top = df[df.Anom_Votes >= 2][["ID", "Campus", "Housing", "Allowance", "SideIncome", "Computed_Total", "Stress", "Anom_Votes"]].sort_values("Anom_Votes", ascending=False)
    top.to_csv(os.path.join(R, "anomalies_consensus.csv"), index=False)
    OUT["anomaly"] = {"votes_ge2": top.to_dict("records"), "counts": {c: int(df[c].sum()) for c in ["Anom_IsoForest", "Anom_LOF", "Anom_Mahal", "Anom_RobustZ"]}}
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=df, x="PC1", y="PC2", hue="Anom_Votes", size="Anom_Score", palette="rocket_r", ax=ax, edgecolor="k")
    for _, r in df[df.Anom_Votes >= 2].iterrows(): ax.annotate(int(r.ID), (r.PC1, r.PC2), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax.set_title("Anomaly detection: consensus of 4 detectors (labels = student ID)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "27_anomalies.png"), dpi=130); plt.close(fig)


# ==================================================================================
# MODEL EVALUATION SUMMARY + BUDGET RECOMMENDER (k-NN peer matching)
# ==================================================================================
def peer_recommender():
    feats = ["Allowance", "SideIncome", "Year"]
    cat = ["Housing", "Campus"]
    X = pd.concat([pd.DataFrame(StandardScaler().fit_transform(df[feats]), columns=feats),
                   pd.get_dummies(df[cat]).astype(float) * 2], axis=1)
    nn = NearestNeighbors(n_neighbors=6).fit(X)
    d, idx = nn.kneighbors(X)
    # leave-one-out: predict each student's cost as the mean of the 5 nearest peers
    pred = np.array([df.loc[i[1:], "Computed_Total"].mean() for i in idx])
    mae = float(np.abs(pred - df.Computed_Total).mean()); r2 = 1 - ((pred - df.Computed_Total) ** 2).sum() / ((df.Computed_Total - df.Computed_Total.mean()) ** 2).sum()
    OUT["peer_recommender"] = {"LOO_MAE": round(mae), "LOO_R2": round(float(r2), 3)}


if __name__ == "__main__":
    regression_models(); classification_models(); clustering_section()
    association_rules_section(); anomaly_section(); peer_recommender()
    df.to_csv(os.path.join(BASE, "data", "v3_model_outputs.csv"), index=False)
    json.dump(OUT, open(os.path.join(R, "part3_results.json"), "w"), indent=2, default=str)
    print(pd.DataFrame(OUT["regression"]["cv_table"]).to_string())
    print(pd.DataFrame(OUT["classification"]["cv_table"]).to_string())
    print(json.dumps({k: v for k, v in OUT["clustering"].items() if k not in ("model_selection",)}, indent=1, default=str)[:4000])
