"""
STUDENT COST OF LIVING INTELLIGENCE SYSTEM
Part 1  |  Units I & III & IV(feature construction)
  - Data Science process (CRISP-DM), data pipeline stages
  - Data quality audit, cleaning, imputation, validation, outlier flags
  - Data versioning, data catalog / dictionary, relational DB (SQLite) + SQL
  - Feature construction
Run:  python part1_pipeline_cleaning.py
"""
import os, json, hashlib, sqlite3, time, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

BASE = os.path.dirname(os.path.abspath(__file__))
def find_raw():
    """Locate the dataset: env var RAW_PATH, then next to the scripts, data/, current folder."""
    cands = [os.environ.get("RAW_PATH", ""),
             os.path.join(BASE, "actualdataset.ods"), os.path.join(BASE, "data", "actualdataset.ods"),
             os.path.join(os.getcwd(), "actualdataset.ods")]
    for c in cands:
        if c and os.path.isfile(c):
            return c
    raise FileNotFoundError("actualdataset.ods not found. Put it in the same folder as these scripts "
                            "(or set RAW_PATH=/full/path/to/actualdataset.ods).")


RAW_PATH = find_raw()
D, F, R, DB = (os.path.join(BASE, x) for x in ("data", "figures", "results", "db"))
for p in (D, F, R, DB):
    os.makedirs(p, exist_ok=True)

STAGE_LOG, CLEAN_LOG = [], []


def stage(name):
    def deco(fn):
        def wrap(*a, **k):
            t = time.time()
            out = fn(*a, **k)
            STAGE_LOG.append({"stage": name, "seconds": round(time.time() - t, 3)})
            print(f"[pipeline] {name:<12} done")
            return out
        return wrap
    return deco


def log(step, detail, rows=None):
    CLEAN_LOG.append({"step": step, "detail": detail, "rows_affected": rows})


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


# ----------------------------------------------------------------- EXTRACT
@stage("extract")
def extract():
    df = pd.read_excel(RAW_PATH, engine="odf")
    df.to_csv(os.path.join(D, "v0_raw.csv"), index=False)
    return df


# ------------------------------------------------------------ QUALITY AUDIT
def quality_audit(df, label):
    n = len(df)
    num = df.select_dtypes("number")
    completeness = 1 - df.isna().sum().sum() / df.size
    uniq = 1 - df.duplicated().mean()
    # validity: non-negative money, stress 1-10, age 16-40, year 1-5
    money = [c for c in df.columns if c.endswith("INR") or c in
             ("Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Reported_Total")]
    bad = 0
    for c in money:
        bad += (df[c] < 0).sum()
    stress_col = [c for c in df.columns if "Stress" in c][0]
    bad += ((df[stress_col] < 1) | (df[stress_col] > 10)).sum()
    validity = 1 - bad / (n * (len(money) + 1))
    # consistency: itemised vs reported total within 25%
    parts = [c for c in df.columns if c in ("Rent_Stay_Est_INR", "Food_Dining_Est_INR",
             "Commute_Travel_Est_INR", "Academics_Lifestyle_Est_INR", "Rent", "Food", "Commute", "Academics")]
    rep = [c for c in df.columns if "Reported_Total" in c][0]
    comp = df[parts].sum(axis=1)
    consistency = ((comp - df[rep]).abs() / df[rep].replace(0, np.nan) <= 0.25).mean()
    # label uniformity: categorical labels that collide after normalisation
    cats = df.select_dtypes(include=["object", "string"]).columns
    collide = 0
    for c in cats:
        s = df[c].dropna().astype(str)
        collide += s.nunique() - s.str.strip().str.lower().nunique()
    uniformity = 1 - collide / max(1, n)
    return {"dataset": label, "rows": n, "completeness": round(completeness, 4),
            "uniqueness": round(uniq, 4), "validity": round(validity, 4),
            "consistency_total_vs_items": round(float(consistency), 4),
            "label_uniformity": round(uniformity, 4)}


# --------------------------------------------------------------------- CLEAN
@stage("clean")
def clean(raw):
    df = raw.copy()
    audit_before = quality_audit(df, "raw")

    # 1 ── tidy names
    df = df.rename(columns={
        "Response_ID": "ID", "Monthly_Allowance_Est_INR": "Allowance",
        "Side_Earning_Est_INR": "SideIncome", "Rent_Stay_Est_INR": "Rent",
        "Food_Dining_Est_INR": "Food", "Commute_Travel_Est_INR": "Commute",
        "Academics_Lifestyle_Est_INR": "Academics",
        "Reported_Total_Spend_Est_INR": "Reported_Total",
        "Financial_Stress_1_10": "Stress", "Academic_Stream": "Stream",
        "Living_Arrangement": "Living", "Area_Locality": "Locality"})
    log("rename", "Standardised 16 column names (short, snake/Title case, units dropped)", 16)

    # 2 ── whitespace + case
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    before = df["Locality"].nunique()
    df["Locality"] = df["Locality"].str.title()
    log("text-normalise",
        f"Trimmed spaces + Title-cased Locality ('bidholi','nanda ki chowki','Prem nagar ' etc.): "
        f"{before} -> {df['Locality'].nunique()} unique values",
        int((raw["Area_Locality"].astype(str) != raw["Area_Locality"].astype(str).str.strip().str.title()).sum()))

    # 3 ── harmonise categories
    n_flat = int((df["Living"] == "Shared Flat / PG").sum())
    df["Living"] = df["Living"].replace({"Shared Flat / PG": "PG (Shared)"})
    log("category-merge", "'Shared Flat / PG' merged into 'PG (Shared)' (same concept, 1 row)", n_flat)

    n_g = int(df["Gender"].isna().sum() + (df["Gender"] == "Prefer not to say").sum())
    df["Gender"] = df["Gender"].fillna("Undisclosed").replace({"Prefer not to say": "Undisclosed"})
    log("category-merge", "Gender: missing + 'Prefer not to say' -> 'Undisclosed' (kept, not dropped)", n_g)

    # 4 ── derived grouping columns (hierarchy used for imputation and analysis)
    housing_map = {"Campus Hostel": "Hostel", "PG (Single)": "PG", "PG (Shared)": "PG",
                   "Rented Flat": "Flat", "Home (Day Scholar)": "Home"}
    df["Housing"] = df["Living"].map(housing_map)
    df["Sharing"] = np.where(df["Living"].isin(["PG (Shared)", "Rented Flat"]), "Shared/Flat",
                     np.where(df["Living"] == "PG (Single)", "Single", "NA"))
    df["Stream_Group"] = df["Stream"].map({
        "B.Tech CSE / AI": "Engineering", "B.Tech Core Engg": "Engineering",
        "Law (BA LLB / BBA LLB)": "Law", "BBA / MBA": "Management",
        "Liberal Studies / Media": "Liberal/Media", "Health Sciences": "Health", "Design": "Design"})
    log("feature-group", "Added Housing (Hostel/PG/Flat/Home) and Stream_Group for stable group statistics", 50)

    # 5 ── missing values: flag first, then impute with group median (robust, n is small)
    for col in ["Allowance", "Commute", "Stress"]:
        miss = df[col].isna()
        df[f"{col}_Imputed"] = miss.astype(int)
        med = df.groupby("Housing")[col].transform("median")
        df[col] = df[col].fillna(med)
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
        log("impute", f"{col}: {int(miss.sum())} missing -> median of same Housing group (+ {col}_Imputed flag)",
            int(miss.sum()))
    df["Stress"] = df["Stress"].round().astype(int)

    # 6 ── types
    for c in ["ID", "Year", "Age"]:
        df[c] = df[c].astype(int)
    for c in ["Allowance", "Commute"]:
        df[c] = df[c].astype(float)
    log("dtype", "Cast ID/Year/Age/Stress to integer; money columns to float", None)

    # 7 ── validity & plausibility checks (flag, never silently delete)
    df["Flag_Age_Year"] = ((df["Age"] < 17 + df["Year"] - 1) | (df["Age"] > 17 + df["Year"] + 3)).astype(int)
    log("validate", "Age vs Year plausibility (expected age ~ 17+Year): flagged implausible rows", int(df["Flag_Age_Year"].sum()))

    df["Computed_Total"] = df[["Rent", "Food", "Commute", "Academics"]].sum(axis=1)
    df["Total_Gap"] = df["Computed_Total"] - df["Reported_Total"]
    df["Flag_Total_Mismatch"] = ((df["Total_Gap"].abs() / df["Reported_Total"].replace(0, np.nan)) > 0.25).astype(int)
    log("validate", "Itemised total vs respondent's Reported_Total differs by >25%: flagged "
        "(Computed_Total, the sum of the 4 items, is used as the modelling target)", int(df["Flag_Total_Mismatch"].sum()))

    df["Flag_Hostel_Fee_Excluded"] = ((df["Housing"] == "Hostel") & (df["Rent"] == 0)).astype(int)
    log("validate", "Hostel students report Rent=0 -> hostel fee is NOT in their cost; flagged so cost "
        "comparisons across housing are interpreted correctly", int(df["Flag_Hostel_Fee_Excluded"].sum()))

    # 8 ── outliers: Tukey fences + robust z (flag, keep)
    out_cols = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Computed_Total"]
    flags = pd.DataFrame(index=df.index)
    for c in out_cols:
        q1, q3 = df[c].quantile([.25, .75])
        iqr = q3 - q1
        flags[c] = ((df[c] < q1 - 1.5 * iqr) | (df[c] > q3 + 1.5 * iqr))
    df["Outlier_Count"] = flags.sum(axis=1)
    df["Flag_Outlier"] = (df["Outlier_Count"] > 0).astype(int)
    log("outliers", "Tukey 1.5*IQR fences on 7 money columns: outlier rows flagged and RETAINED "
        "(genuine high-income / side-earning students, not errors)", int(df["Flag_Outlier"].sum()))

    # 9 ── duplicates
    log("duplicates", f"Exact duplicate rows: {int(df.duplicated(subset=[c for c in df.columns if c != 'ID']).sum())}; "
        f"duplicate IDs: {int(df['ID'].duplicated().sum())}", 0)

    audit_after = quality_audit(df.rename(columns={}), "cleaned")
    df["Gender"] = df["Gender"].astype(str)
    return df, [audit_before, audit_after], flags


# --------------------------------------------------------- FEATURE CONSTRUCTION
@stage("wrangle")
def construct_features(df):
    f = df.copy()
    f["Total_Income"] = f["Allowance"] + f["SideIncome"]
    f["Net_Balance"] = f["Total_Income"] - f["Computed_Total"]
    f["Savings_Rate"] = f["Net_Balance"] / f["Total_Income"]
    f["Spend_to_Income"] = f["Computed_Total"] / f["Total_Income"]
    f["Deficit"] = (f["Net_Balance"] < 0).astype(int)
    f["Self_Funded_Share"] = f["SideIncome"] / f["Total_Income"]
    for c in ["Rent", "Food", "Commute", "Academics"]:
        f[f"{c}_Share"] = f[c] / f["Computed_Total"]
    f["Essential_Spend"] = f["Rent"] + f["Food"] + f["Commute"]
    f["Essential_Ratio"] = f["Essential_Spend"] / f["Computed_Total"]
    f["Log_Total"] = np.log1p(f["Computed_Total"])
    f["Log_Income"] = np.log1p(f["Total_Income"])
    f["Has_Side_Income"] = (f["SideIncome"] > 0).astype(int)
    f["Is_Offcampus_Paying"] = (f["Rent"] > 0).astype(int)
    f["High_Stress"] = (f["Stress"] >= 7).astype(int)
    f["Stress_Level"] = pd.cut(f["Stress"], [0, 3, 6, 10], labels=["Low", "Medium", "High"])
    f["Allowance_Tier"] = pd.qcut(f["Allowance"], 3, labels=["Low", "Mid", "High"])
    # Student Cost Index: 100 = median student's itemised monthly cost
    f["Cost_Index"] = 100 * f["Computed_Total"] / f["Computed_Total"].median()
    # Financial Wellbeing Score (0-100): 50% cash cushion, 50% low stress
    cushion = f["Savings_Rate"].clip(-1, 1)
    f["Wellbeing_Score"] = 100 * (0.5 * (cushion + 1) / 2 + 0.5 * (10 - f["Stress"]) / 9)
    f["Stress_Level"] = f["Stress_Level"].astype(str)
    f["Allowance_Tier"] = f["Allowance_Tier"].astype(str)
    return f


# ---------------------------------------------------------------- DB + SQL
def build_database(f):
    path = os.path.join(DB, "student_cost.db")
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE student(
        id INTEGER PRIMARY KEY, campus TEXT, stream TEXT, stream_group TEXT, year INT, age INT, gender TEXT,
        living TEXT, housing TEXT, locality TEXT, allowance REAL, side_income REAL, stress INT);
    CREATE TABLE expense(
        id INTEGER, category TEXT, amount REAL, FOREIGN KEY(id) REFERENCES student(id));
    CREATE TABLE locality_dim(locality TEXT PRIMARY KEY, campus TEXT);
    CREATE INDEX ix_exp ON expense(id, category);
    """)
    s = f[["ID", "Campus", "Stream", "Stream_Group", "Year", "Age", "Gender", "Living", "Housing",
           "Locality", "Allowance", "SideIncome", "Stress"]]
    cur.executemany("INSERT INTO student VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", s.values.tolist())
    long = f.melt(id_vars="ID", value_vars=["Rent", "Food", "Commute", "Academics"],
                  var_name="category", value_name="amount")
    cur.executemany("INSERT INTO expense VALUES (?,?,?)", long.values.tolist())
    loc = f.groupby("Locality")["Campus"].agg(lambda x: x.mode()[0]).reset_index()
    cur.executemany("INSERT INTO locality_dim VALUES (?,?)", loc.values.tolist())
    con.commit()

    queries = {
        "Q1  Average monthly cost by housing type (JOIN + GROUP BY)":
            """SELECT s.housing, COUNT(DISTINCT s.id) n, ROUND(SUM(e.amount)*1.0/COUNT(DISTINCT s.id)) avg_cost
               FROM student s JOIN expense e USING(id) GROUP BY s.housing ORDER BY avg_cost DESC""",
        "Q2  Category cost per campus (pivot via CASE)":
            """SELECT s.campus,
                      ROUND(AVG(CASE WHEN e.category='Rent' THEN e.amount END)) rent,
                      ROUND(AVG(CASE WHEN e.category='Food' THEN e.amount END)) food,
                      ROUND(AVG(CASE WHEN e.category='Commute' THEN e.amount END)) commute,
                      ROUND(AVG(CASE WHEN e.category='Academics' THEN e.amount END)) academics
               FROM student s JOIN expense e USING(id) GROUP BY s.campus""",
        "Q3  Costliest localities with at least 3 students (HAVING)":
            """SELECT s.locality, COUNT(DISTINCT s.id) n, ROUND(SUM(e.amount)*1.0/COUNT(DISTINCT s.id)) avg_cost
               FROM student s JOIN expense e USING(id) GROUP BY s.locality HAVING n>=3 ORDER BY avg_cost DESC""",
        "Q4  Students spending more than they receive (subquery)":
            """SELECT s.id, s.campus, s.housing, s.allowance+s.side_income income, t.spend,
                      t.spend-(s.allowance+s.side_income) shortfall
               FROM student s JOIN (SELECT id, SUM(amount) spend FROM expense GROUP BY id) t USING(id)
               WHERE t.spend > s.allowance+s.side_income ORDER BY shortfall DESC""",
        "Q5  Stress by year of study (aggregate)":
            "SELECT year, COUNT(*) n, ROUND(AVG(stress),2) avg_stress, MAX(stress) max_stress FROM student GROUP BY year",
        "Q6  Rank students within housing by spend (window function)":
            """SELECT * FROM (SELECT s.id, s.housing, SUM(e.amount) spend,
                      RANK() OVER (PARTITION BY s.housing ORDER BY SUM(e.amount) DESC) rk
               FROM student s JOIN expense e USING(id) GROUP BY s.id) WHERE rk<=2 ORDER BY housing, rk""",
    }
    md = ["# SQL queries on `student_cost.db` (Unit III: working with relational databases)\n"]
    for name, q in queries.items():
        res = pd.read_sql_query(q, con)
        md.append(f"## {name}\n```sql\n{' '.join(q.split())}\n```\n\n{res.to_markdown(index=False)}\n")
    open(os.path.join(R, "sql_query_results.md"), "w").write("\n".join(md))
    con.close()
    return path


# -------------------------------------------------------------------- CATALOG
def build_catalog(raw, clean_df, feat, audits):
    dictionary = {
        "ID": "Respondent identifier", "Campus": "Bidholi or Kandoli campus",
        "Stream": "Programme as reported", "Stream_Group": "Programme family (derived)",
        "Year": "Year of study 1-4", "Age": "Years", "Gender": "Male/Female/Undisclosed",
        "Living": "Living arrangement (cleaned)", "Housing": "Hostel/PG/Flat/Home (derived)",
        "Locality": "Area where student lives (cleaned)",
        "Allowance": "Monthly allowance from family (INR)", "SideIncome": "Monthly side earning (INR)",
        "Rent": "Monthly rent / stay (INR); 0 for hostel & home -> hostel fee not captured",
        "Food": "Monthly food & dining (INR)", "Commute": "Monthly travel (INR)",
        "Academics": "Monthly academics & lifestyle (INR)",
        "Reported_Total": "Respondent's own total estimate (INR)",
        "Computed_Total": "Sum of the four itemised costs (INR) - modelling target",
        "Stress": "Self-rated financial stress 1(low)-10(high)",
        "Total_Income": "Allowance + SideIncome", "Net_Balance": "Total_Income - Computed_Total",
        "Savings_Rate": "Net_Balance / Total_Income", "Spend_to_Income": "Computed_Total / Total_Income",
        "Deficit": "1 if spending exceeds income", "High_Stress": "1 if Stress >= 7",
        "Cost_Index": "100 x cost relative to median student", "Wellbeing_Score": "0-100 composite of cash cushion and low stress",
        "Flag_*": "Data-quality flags (kept, not removed)"}
    files = {n: sha(os.path.join(D, n)) for n in sorted(os.listdir(D)) if n.endswith(".csv")}
    cat = {"project": "Student Cost of Living Intelligence System", "source_file": os.path.basename(RAW_PATH),
           "versions": {"v0_raw": "as uploaded", "v1_clean": "cleaned + flagged", "v2_features": "clean + engineered features"},
           "sha256_prefix": files, "quality_audit": audits, "pipeline_stages": STAGE_LOG,
           "cleaning_log": CLEAN_LOG, "data_dictionary": dictionary, "rows": len(feat), "columns": feat.shape[1]}
    json.dump(cat, open(os.path.join(R, "data_catalog.json"), "w"), indent=2, default=str)
    pd.DataFrame(CLEAN_LOG).to_csv(os.path.join(R, "cleaning_log.csv"), index=False)
    pd.DataFrame(audits).to_csv(os.path.join(R, "quality_audit.csv"), index=False)


# ------------------------------------------------------------------ PLOTS (EDA)
def eda_plots(raw, df, f, flags):
    # missingness
    fig, ax = plt.subplots(figsize=(8, 3.5))
    miss = raw.isna().sum()
    miss = miss[miss > 0]
    ax.barh(miss.index, miss.values, color="#e07a5f")
    ax.set_title("Missing values in raw data"); ax.set_xlabel("count")
    fig.tight_layout(); fig.savefig(os.path.join(F, "01_missing_values.png"), dpi=130); plt.close(fig)

    # before/after quality
    # distributions
    cols = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Computed_Total", "Stress"]
    fig, axs = plt.subplots(2, 4, figsize=(15, 6))
    for a, c in zip(axs.ravel(), cols):
        sns.histplot(f[c], kde=True, ax=a, color="#3d405b"); a.set_title(c)
    fig.suptitle("Univariate distributions (cleaned data)", y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(F, "02_distributions.png"), dpi=130); plt.close(fig)

    # outliers
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=f[["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Computed_Total"]],
                orient="h", ax=ax, palette="Set2")
    ax.set_title("Outlier check (Tukey fences) - flagged, retained")
    fig.tight_layout(); fig.savefig(os.path.join(F, "03_outliers_boxplot.png"), dpi=130); plt.close(fig)

    # stacked spend by housing
    order = ["Home", "Hostel", "Flat", "PG"]
    g = f.groupby("Housing")[["Rent", "Food", "Commute", "Academics"]].mean().loc[order]
    ax = g.plot(kind="bar", stacked=True, figsize=(8, 5), colormap="Set2", rot=0)
    ax.set_ylabel("Mean monthly INR"); ax.set_title("Where the money goes, by housing type\n(hostel fee not captured in survey)")
    plt.tight_layout(); plt.savefig(os.path.join(F, "04_spend_breakdown_housing.png"), dpi=130); plt.close()

    # correlation heat map
    num = ["Allowance", "SideIncome", "Rent", "Food", "Commute", "Academics", "Computed_Total",
           "Total_Income", "Net_Balance", "Savings_Rate", "Stress", "Year", "Age"]
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    sns.heatmap(f[num].corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation matrix")
    fig.tight_layout(); fig.savefig(os.path.join(F, "05_correlation_heatmap.png"), dpi=130); plt.close(fig)

    # locality
    loc = f.groupby("Locality")["Computed_Total"].agg(["mean", "count"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([f"{i} (n={r['count']})" for i, r in loc.iterrows()], loc["mean"], color="#81b29a")
    ax.set_title("Average monthly cost by locality"); ax.set_xlabel("INR")
    fig.tight_layout(); fig.savefig(os.path.join(F, "06_locality_cost.png"), dpi=130); plt.close(fig)

    # income vs spend, coloured by stress
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sc = ax.scatter(f["Total_Income"], f["Computed_Total"], c=f["Stress"], cmap="YlOrRd", s=70, edgecolor="k")
    lim = max(f["Total_Income"].max(), f["Computed_Total"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="spend = income")
    ax.set_xlabel("Total income (INR)"); ax.set_ylabel("Itemised spend (INR)")
    plt.colorbar(sc, label="Financial stress"); ax.legend(); ax.set_title("Income vs spend (colour = stress)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "07_income_vs_spend.png"), dpi=130); plt.close(fig)

    # stress by housing / year
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.boxplot(data=f, x="Housing", y="Stress", order=["Home", "Hostel", "Flat", "PG"], ax=axs[0], palette="Pastel1")
    sns.boxplot(data=f, x="Year", y="Stress", ax=axs[1], palette="Pastel2")
    fig.suptitle("Financial stress patterns"); fig.tight_layout()
    fig.savefig(os.path.join(F, "08_stress_patterns.png"), dpi=130); plt.close(fig)

    # reported vs computed
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(f["Reported_Total"], f["Computed_Total"], c=np.where(f["Flag_Total_Mismatch"] == 1, "#e63946", "#457b9d"), s=60)
    lim = max(f["Reported_Total"].max(), f["Computed_Total"].max())
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("Reported total"); ax.set_ylabel("Sum of itemised costs"); ax.set_title("Consistency check (red = >25% mismatch)")
    fig.tight_layout(); fig.savefig(os.path.join(F, "09_reported_vs_computed.png"), dpi=130); plt.close(fig)


# ---------------------------------------------------------------------- MAIN
def run():
    raw = extract()
    cleaned, audits, flags = clean(raw)
    feat = construct_features(cleaned)
    for c in feat.select_dtypes(include=["string"]).columns:
        feat[c] = feat[c].astype(str)
    cleaned.drop(columns=[]).to_csv(os.path.join(D, "v1_clean.csv"), index=False)
    feat.to_csv(os.path.join(D, "v2_features.csv"), index=False)
    build_database(feat)
    eda_plots(raw, cleaned, feat, flags)
    build_catalog(raw, cleaned, feat, audits)
    with pd.ExcelWriter(os.path.join(D, "student_cost_clean.xlsx")) as w:
        feat.to_excel(w, sheet_name="clean_with_features", index=False)
        pd.DataFrame(CLEAN_LOG).to_excel(w, sheet_name="cleaning_log", index=False)
        pd.DataFrame(audits).to_excel(w, sheet_name="quality_audit", index=False)
    print(pd.DataFrame(audits).T)
    print(pd.DataFrame(CLEAN_LOG)[["step", "detail", "rows_affected"]].to_string())
    return feat


if __name__ == "__main__":
    run()
