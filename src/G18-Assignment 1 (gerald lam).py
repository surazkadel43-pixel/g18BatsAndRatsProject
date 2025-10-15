import os
import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt

# Load Datasets
try:
    df1 = pd.read_csv("Data/dataset1.csv")
    df2 = pd.read_csv("Data/dataset2.csv")
except Exception:
    df1 = pd.read_csv("Data/dataset1.csv")
    df2 = pd.read_csv("Data/dataset2.csv")

sns.set_theme(style="whitegrid")

def parse_ddmmyy_hm(s):
    # some timestamps are dd/mm/yyyy; use both %H:%M and %H:%M:%S
    x = pd.to_datetime(s, format="%d/%m/%Y %H:%M", dayfirst=True, errors="coerce")
    if isinstance(s, pd.Series) and x.isna().any():
        y = pd.to_datetime(s, format="%d/%m/%Y %H:%M:%S", dayfirst=True, errors="coerce")
        x = x.fillna(y)
    return x

def normalise_month_to_int(m):
    # setting months 1..12; accept ints or short names
    if pd.isna(m):
        return np.nan
    try:
        v = int(m)
        return v if 1 <= v <= 12 else np.nan
    except Exception:
        s = str(m).strip().lower()[:3]
        mapping = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
                   'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12}
        return mapping.get(s, np.nan)

def month_to_season_egypt(month_int):
    # Egypt seasons for Investigation B
    if pd.isna(month_int):
        return np.nan
    m = int(month_int)
    if m in [12,1,2]: return "winter"
    if m in [3,4,5]:  return "spring"
    return "other"

def binary_clean(x):
    # force to 0/1 if already binary; otherwise median-threshold
    s = pd.to_numeric(x, errors="coerce")
    if s.dropna().isin([0,1]).all():
        return s.fillna(0).astype(int)
    med = s.median(skipna=True)
    return (s >= med).astype(int)

def series_mode_first(s):
    m = s.mode(dropna=True)
    return m.iloc[0] if len(m) else np.nan

def iqr(s):
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    return q3 - q1

def print_ols_minimal(model, title="OLS results"):
    print(f"\n[{title}]")
    params = model.params
    tvals = model.tvalues
    pvals = model.pvalues
    for name in params.index:
        print(f"{name:>28}: coef={params[name]:+.6f}  t={tvals[name]:+.3f}  p={pvals[name]:.6g}")

# Data Cleaning

for c in ["start_time", "rat_period_start", "rat_period_end", "sunset_time"]:
    if c in df1.columns:
        df1[c] = parse_ddmmyy_hm(df1[c]) if df1[c].dtype == object else pd.to_datetime(df1[c], errors="coerce")

# df2 time
if "time" in df2.columns:
    df2["time"] = parse_ddmmyy_hm(df2["time"])
    df2["period_start"] = df2["time"].dt.floor("30min")

# months → int; if missing, try derive from timestamp
if "month" in df1.columns:
    df1["month_int"] = df1["month"].apply(normalise_month_to_int)
else:
    df1["month_int"] = np.nan
if df1["month_int"].isna().any() and "start_time" in df1.columns:
    df1.loc[df1["month_int"].isna(), "month_int"] = df1["start_time"].dt.month

if "month" in df2.columns:
    df2["month_int"] = df2["month"].apply(normalise_month_to_int)
else:
    df2["month_int"] = np.nan
if df2["month_int"].isna().any() and "time" in df2.columns:
    df2.loc[df2["month_int"].isna(), "month_int"] = df2["time"].dt.month

# Egypt seasons
df1["season"] = df1["month_int"].apply(month_to_season_egypt)
df2["season"] = df2["month_int"].apply(month_to_season_egypt)

# Restrict season work to ONLY Winter/Spring for season-based analysis/plots
VALID_SEASONS = ["winter", "spring"]
df1["season"] = df1["season"].where(df1["season"].isin(VALID_SEASONS))
df2["season"] = df2["season"].where(df2["season"].isin(VALID_SEASONS))

# Numeric casts
for c in ["seconds_after_rat_arrival", "bat_landing_to_food", "hours_after_sunset", "risk", "reward"]:
    if c in df1.columns:
        if c in ["risk","reward"]:
            df1[c] = binary_clean(df1[c])
        else:
            df1[c] = pd.to_numeric(df1[c], errors="coerce")

for c in ["hours_after_sunset","bat_landing_number","food_availability","rat_minutes","rat_arrival_number"]:
    if c in df2.columns:
        df2[c] = pd.to_numeric(df2[c], errors="coerce")

# Simple median impute for numeric columns
for d in (df1, df2):
    for c in d.select_dtypes(include=[np.number]).columns:
        if d[c].isna().any():
            d[c] = d[c].fillna(d[c].median())

# Season-filtered frames used for all season-based analysis/plots
df1_B = df1[df1["season"].isin(VALID_SEASONS)].copy()
df2_B = df2[df2["season"].isin(VALID_SEASONS)].copy()

print("df1_B seasons (Win/Spr only):", df1_B["season"].value_counts(dropna=False).to_dict())
print("df2_B seasons (Win/Spr only):", df2_B["season"].value_counts(dropna=False).to_dict())

# Seasonal differences in rat activity (Student's/pooled t-tests)
if {"rat_arrival_number","season"}.issubset(df2_B.columns):
    w = df2_B.loc[df2_B["season"]=="winter","rat_arrival_number"]
    s = df2_B.loc[df2_B["season"]=="spring","rat_arrival_number"]
    t, p = st.ttest_ind(w, s, equal_var=True)  # <-- normal t-test
    print("\n[Part B] Rat arrivals: Winter vs Spring (df2)")
    print(f"winter mean={w.mean():.2f} (n={w.size}), spring mean={s.mean():.2f} (n={s.size})")
    print(f"t={t:.3f}, p(two-sided)={p:.6f}")

if {"rat_minutes","season"}.issubset(df2_B.columns):
    w = df2_B.loc[df2_B["season"]=="winter","rat_minutes"]
    s = df2_B.loc[df2_B["season"]=="spring","rat_minutes"]
    t, p = st.ttest_ind(w, s, equal_var=True)  # <-- normal t-test
    print("\n[Part B] Rat minutes: Winter vs Spring (df2)")
    print(f"winter mean={w.mean():.2f} (n={w.size}), spring mean={s.mean():.2f} (n={s.size})")
    print(f"t={t:.3f}, p(two-sided)={p:.6f}")

# How rat time predicts bat landings (df2_B)
if {"bat_landing_number","rat_minutes"}.issubset(df2_B.columns):
    mB3 = smf.ols("bat_landing_number ~ rat_minutes", data=df2_B).fit()
    print_ols_minimal(mB3, "Part B / Simple OLS (df2 Win/Spr): bat_landing_number ~ rat_minutes")

need2 = ["bat_landing_number","rat_minutes","food_availability","hours_after_sunset","season"]
if set(need2).issubset(df2_B.columns):
    mB4 = smf.ols(
        "bat_landing_number ~ rat_minutes + food_availability + hours_after_sunset + C(season)",
        data=df2_B
    ).fit()
    print_ols_minimal(mB4, "Part B / Multiple OLS (df2 Win/Spr): bat_landing_number ~ predictors + C(season)")

# Visualisation

# Histogram: seconds_after_rat_arrival (df1) — not season-specific
if "seconds_after_rat_arrival" in df1.columns:
    plt.figure(figsize=(6,4))
    plt.hist(df1["seconds_after_rat_arrival"], bins=20)
    plt.xlabel("Seconds since rat arrival")
    plt.ylabel("Number of landings")
    plt.title("Distribution of seconds_after_rat_arrival (df1)")
    plt.tight_layout()
    plt.show()

# Bat landings by season (df2_B) — only winter/spring
if {"bat_landing_number","season"}.issubset(df2_B.columns):
    plt.figure(figsize=(6,4))
    sns.barplot(
        x="season", y="bat_landing_number", data=df2_B,
        order=["winter","spring"], errorbar=("ci", 95)
    )
    plt.title("Bat landings per 30 min by Egypt season (df2)")
    plt.ylabel("Mean landings")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# Food availability by season (df2_B)
if {"food_availability","season"}.issubset(df2_B.columns):
    plt.figure(figsize=(6,4))
    sns.boxplot(x="season", y="food_availability", data=df2_B, order=["winter","spring"])
    plt.title("Food availability by Egypt season (df2)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# Rat minutes by season (df2_B) — violin
if {"rat_minutes","season"}.issubset(df2_B.columns):
    plt.figure(figsize=(6,4))
    sns.violinplot(x="season", y="rat_minutes", data=df2_B, order=["winter","spring"], inner="box")
    plt.title("Rat minutes per 30 min by Egypt season (df2)")
    plt.xlabel("")
    plt.tight_layout()
    plt.show()

# Scatter with regression line: Rat minutes vs Bat landings, coloured by season (df2_B)
if {"rat_minutes","bat_landing_number","season"}.issubset(df2_B.columns):
    g = sns.lmplot(
        x="rat_minutes", y="bat_landing_number", hue="season", data=df2_B,
        order=1, ci=95, height=4, aspect=1.3, scatter_kws=dict(alpha=0.6),
        hue_order=["winter","spring"]
    )
    g.fig.suptitle("Rat minutes vs Bat landings by season (df2)", y=1.02)
    plt.show()

# Monthly trend across months (df2_B only, so Win/Spr months)
if {"month_int","bat_landing_number"}.issubset(df2_B.columns):
    plt.figure(figsize=(7,4))
    sns.lineplot(
        x="month_int", y="bat_landing_number", data=df2_B,
        estimator="mean", errorbar=("ci", 95), marker="o"
    )
    plt.xticks(range(1,13))
    plt.xlabel("Month (1–12)")
    plt.ylabel("Mean bat landings per 30 min")
    plt.title("Monthly trend of bat landings (Win/Spr only, df2)")
    plt.tight_layout()
    plt.show()

# Descriptive Statistics

def print_descriptives(label, s):
    s_clean = pd.to_numeric(s, errors="coerce").dropna()
    print(f"\n[Descriptives] {label}")
    print(f"n={len(s_clean)}  mean={s_clean.mean():.3f}  median={s_clean.median():.3f}  mode={series_mode_first(s_clean)}")
    print(f"variance={s_clean.var(ddof=1):.3f}  std={s_clean.std(ddof=1):.3f}  IQR={iqr(s_clean):.3f}")

# Overall timing descriptives
if "seconds_after_rat_arrival" in df1.columns:
    print_descriptives("df1.seconds_after_rat_arrival", df1["seconds_after_rat_arrival"])

# Season-wise descriptives for df2_B
if {"bat_landing_number","season"}.issubset(df2_B.columns):
    for ssn in ["winter","spring"]:
        print_descriptives(f"df2_B.bat_landing_number ({ssn})", df2_B.loc[df2_B["season"]==ssn,"bat_landing_number"])

if {"rat_minutes","season"}.issubset(df2_B.columns):
    for ssn in ["winter","spring"]:
        print_descriptives(f"df2_B.rat_minutes ({ssn})", df2_B.loc[df2_B["season"]==ssn,"rat_minutes"])

# Hypothesis Test 1 (Investigation A) — Student's two-sample t-test
# Do bats take fewer risks right after a rat arrives?
if "seconds_after_rat_arrival" in df1.columns and "risk" in df1.columns:
    seconds = pd.to_numeric(df1["seconds_after_rat_arrival"], errors="coerce")
    cutoff = seconds.median()
    print("\nMedian of seconds_after_rat_arrival:", cutoff)

    df1["timing"] = np.where(seconds <= cutoff, "early", "late")
    df1["risk_bin"] = binary_clean(df1["risk"])

    early_risk = df1.loc[df1["timing"] == "early", "risk_bin"]
    late_risk  = df1.loc[df1["timing"] == "late",  "risk_bin"]

    print("Early mean risk:", early_risk.mean())
    print("Late  mean risk:", late_risk.mean())
    print("n early:", early_risk.size, " n late:", late_risk.size)

    # normal t-test (pooled variance), one-sided "less"
    t_stat, p_one = st.ttest_ind(early_risk, late_risk, equal_var=True, alternative="less")
    print("\n[Part A: Landing-level test — early vs late risk]")
    print("t-statistic:", t_stat)
    print("p-value (one-sided):", p_one)

# Hypothesis Test 2 (Investigation A) — Student's two-sample t-test
# Do bats land less often during periods when rats are present? (df2)
if set(["period_start","bat_landing_number","rat_minutes"]).issubset(df2.columns):
    d = (
        df2[["period_start","bat_landing_number","rat_minutes"]]
        .dropna(subset=["bat_landing_number","rat_minutes"])
        .drop_duplicates("period_start")
    )
    d["rat_present"] = d["rat_minutes"] > 0

    land_present = d.loc[d["rat_present"], "bat_landing_number"]
    land_absent  = d.loc[~d["rat_present"], "bat_landing_number"]

    print(f"\n[Part A: Period-level test — rat-present vs rat-absent]")
    print(f"Periods: total={len(d)}, rat_present={land_present.size}, rat_absent={land_absent.size}")
    print(f"Means → present={land_present.mean():.2f}, absent={land_absent.mean():.2f}")

    # normal t-test (pooled variance), one-sided "less"
    t_stat2, p_one2 = st.ttest_ind(land_present, land_absent, equal_var=True, alternative="less")
    print(f"t-statistic: {t_stat2:.3f}, p(one-sided)={p_one2:.6f}")
else:
    print("\n[Part A: Period-level test skipped] Missing columns in dataset2.")

# Investigation A — Regressions

# Simple OLS: reward ~ risk  (df1)
if {"reward","risk"}.issubset(df1.columns):
    mA1 = smf.ols("reward ~ risk", data=df1).fit()
    print_ols_minimal(mA1, "Part A / Simple OLS: reward ~ risk (all data)")

# Multiple OLS: reward ~ risk + hours_after_sunset + bat_landing_to_food + seconds_after_rat_arrival
cand = ["reward","risk","hours_after_sunset","bat_landing_to_food","seconds_after_rat_arrival"]
if set(cand).issubset(df1.columns):
    mA2 = smf.ols("reward ~ risk + hours_after_sunset + bat_landing_to_food + seconds_after_rat_arrival", data=df1).fit()
    print_ols_minimal(mA2, "Part A / Multiple OLS: reward ~ risk + timing/context (all data)")

# Investigation B — Winter vs Spring (df1_B)

# Risk difference (Winter vs Spring) — Student's t-test (two-sided)
if "risk" in df1_B.columns:
    wr = df1_B.loc[df1_B["season"]=="winter","risk"]
    sr = df1_B.loc[df1_B["season"]=="spring","risk"]
    t, p = st.ttest_ind(wr, sr, equal_var=True)  # <-- normal t-test
    print("\n[Part B] Risk: Winter vs Spring (df1)")
    print(f"winter mean={wr.mean():.3f} (n={wr.size}), spring mean={sr.mean():.3f} (n={sr.size})")
    print(f"t={t:.4f}, p(two-sided)={p:.6f}")

# Reward difference (Winter vs Spring) — Student's t-test (two-sided)
if "reward" in df1_B.columns:
    wq = df1_B.loc[df1_B["season"]=="winter","reward"]
    sq = df1_B.loc[df1_B["season"]=="spring","reward"]
    t, p = st.ttest_ind(wq, sq, equal_var=True)  # <-- normal t-test
    print("\n[Part B] Reward: Winter vs Spring (df1)")
    print(f"winter mean={wq.mean():.3f} (n={wq.size}), spring mean={sq.mean():.3f} (n={sq.size})")
    print(f"t={t:.4f}, p(two-sided)={p:.6f}")

# Simple OLS (Win/Spr): reward ~ risk
if {"reward","risk"}.issubset(df1_B.columns):
    mB1 = smf.ols("reward ~ risk", data=df1_B).fit()
    print_ols_minimal(mB1, "Part B / Simple OLS (Win/Spr): reward ~ risk")

# Multiple OLS (Win/Spr): add seasonal dummy + timing/context
need = ["reward","risk","hours_after_sunset","bat_landing_to_food","seconds_after_rat_arrival","season"]
if set(need).issubset(df1_B.columns):
    mB2 = smf.ols(
        "reward ~ risk + hours_after_sunset + bat_landing_to_food + seconds_after_rat_arrival + C(season)",
        data=df1_B
    ).fit()
    print_ols_minimal(mB2, "Part B / Multiple OLS (Win/Spr): reward ~ predictors + C(season)")
