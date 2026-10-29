### Import delle librerie
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import StratifiedKFold, GridSearchCV

import joblib
from tqdm import tqdm

from sklearn.tree import plot_tree

plt.style.use("fivethirtyeight")


class tqdm_joblib:
    def __init__(self, total, desc="GridSearchCV"):
        self.pbar = tqdm(total=total, desc=desc)
        self.old_cb = joblib.parallel.BatchCompletionCallBack
        pbar = self.pbar
        class CB(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                pbar.update(n=self.batch_size)
                super().__call__(*args, **kwargs)
        joblib.parallel.BatchCompletionCallBack = CB
    def __enter__(self): return self
    def __exit__(self, *args):
        joblib.parallel.BatchCompletionCallBack = self.old_cb
        self.pbar.close()


def logToFile(data, output):
    print(data)
    if output != None:
        with open(output, 'a', encoding='utf-8') as file:
            file.write(str(data) + '\n')

random_seed = 42

import matplotlib

matplotlib.use("Agg")  # Disattiva la GUI di Tkinter per evitare crash con la parallelizzazione

### DICHIARAZIONE FILE DI LOG E PLOT
pathDebug = 'data/debug/log.txt'
pathPlots = 'data/plots/'

### Pulizia file di log

if os.path.exists(pathDebug):
    os.remove(pathDebug)

### Import dei file

path_file = r'Student_Depression_Dataset.csv'
df = pd.read_csv(path_file)


### Analisi missing values
colMissing = df.isna().sum().to_frame(name='Valori mancanti')
colMissing = tabulate(colMissing, headers="keys", tablefmt='rounded_grid')

logToFile('\n### ANALISI DATI MANCANTI', pathDebug)
logToFile(colMissing, pathDebug)


stats = df.describe()
stats = tabulate(stats, headers="keys", tablefmt="rounded_grid", floatfmt=".2f")

logToFile('\n### Analisi statistica colonne numeriche', pathDebug)
logToFile(stats, pathDebug)


# Coerenza logica: relazione e stato civile
cross_rel_mar = pd.crosstab(df["Relationship_status"], df["Marital_status"])
logToFile('\n### RELATIONSHIP STATUS VS MARITAL STATUS', pathDebug)
logToFile(tabulate(cross_rel_mar, headers="keys", tablefmt="rounded_grid"), pathDebug)


### PLOT Age Distribution by Gender ### 

fig, ax1 = plt.subplots(figsize=(10, 6))

# Asse sinistro: conteggi assoluti
sns.histplot(
    data=df,
    x="Age",
    hue="Gender",
    discrete=True,
    multiple="dodge",
    shrink=0.85,
    stat="count",
    ax=ax1
)

ax1.set_title("Age Distribution by Gender")
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of Students")

# Asse destro: densità
ax2 = ax1.twinx()

sns.histplot(
    data=df,
    x="Age",
    hue="Gender",
    discrete=True,
    multiple="dodge",
    shrink=0.85,
    stat="density",
    alpha=1,
    ax=ax2
)
ax2.grid(False)
ax2.set_ylabel("Density")

plt.tight_layout()

plt.savefig(pathPlots+"plot_AgeDistribution_Gender.png", format="png", bbox_inches="tight")


### PLOT Depression status by studends ### 

depression_counts = df["Depression"].value_counts()

depression_pct = (
    df["Depression"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

depression_summary = pd.DataFrame({
    "Count": depression_counts,
    "Percentage": depression_pct
})

fig, ax = plt.subplots(figsize=(7, 6))

ax = sns.countplot(
    data=df,
    x="Depression"
)

# Etichette con conteggio e percentuale
for container in ax.containers:
    labels = []

    for bar in container:
        height = bar.get_height()
        percentage = height / len(df) * 100
        labels.append(f"{int(height)}\n({percentage:.1f}%)")

    ax.bar_label(container, labels=labels, padding=3)

ax.set_title("Reported Depression Status Among Students", fontsize=14, pad=12)
ax.set_xlabel("Depression Status")
ax.set_ylabel("Number of Students")
ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

plt.tight_layout()
plt.savefig(pathPlots+"plot_DepressionStatus.png", format="png", bbox_inches="tight")


### Plot depressione in base a categoria

def plot_depression_comparison(column, title=None):
    title = title or column.replace("_", " ").title()
    counts = pd.crosstab(df[column], df["Depression"]).reindex(
        columns=["No", "Yes"], fill_value=0
    )
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    order = counts.sum(axis=1).sort_values(ascending=False).index
    counts, pct = counts.loc[order], pct.loc[order]

    ax = counts.plot(
        kind="bar", stacked=True,
        figsize=(max(8, len(order) * 1.2), 6),
        color=["tab:blue", "tab:orange"]
    )

    for i, container in enumerate(ax.containers):
        labels = [
            f"{n}\n({p:.1f}%)" if n > 0 else ""
            for n, p in zip(counts.iloc[:, i], pct.iloc[:, i])
        ]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=12)

    ax.set(title=f"{title} and Depression",
           xlabel=title, ylabel="Number of Students")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Depression")
    ax.grid(axis="y", alpha=0.8)
    plt.savefig(pathPlots+"plot_"+column+"_vs_Depression.png", format="png", bbox_inches="tight")

plot_depression_comparison("Family_background")
plot_depression_comparison("Relationship_status")
plot_depression_comparison("Marital_status")
plot_depression_comparison("Living_style")
plot_depression_comparison("Health_condition")
plot_depression_comparison("friend_circle")

### STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE

summary_list = []
categorical_columns = [
    col for col in df.select_dtypes(include=["object", "category", "string"]).columns
    if col != "Depression"
]

for col in categorical_columns:
    summary = df.groupby(col, observed=True)["Depression"].apply(
        lambda x: (x == "Yes").mean() * 100
    ).reset_index(name="Depression_Rate")

    summary["Variable"] = col
    summary = summary.rename(columns={col: "Category"})
    summary_list.append(summary)

depression_rates = pd.concat(summary_list, ignore_index=True)

depression_rates = depression_rates[
    ["Variable", "Category", "Depression_Rate"]
].sort_values("Depression_Rate", ascending=False)

depression_rates["Depression_Rate"] = depression_rates["Depression_Rate"].round(2)
depression_rates = tabulate(depression_rates, headers="keys", tablefmt="rounded_grid", floatfmt=".2f")
logToFile('\n### STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE', pathDebug)
logToFile(depression_rates, pathDebug)


### TEST CHI QUADRO E V-CRAMER

association_results = []

for col in categorical_columns:
    table = pd.crosstab(df[col], df["Depression"])
    chi2, p, dof, expected = chi2_contingency(table)
    n, r, k = table.to_numpy().sum(), *table.shape
    v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else np.nan

    association_results.append({
        "Variable": col, "Chi-square": round(chi2, 2),
        "Degrees of Freedom": dof,
        "P-value": "< 0.001" if p < 0.001 else f"{p:.3f}",
        "Cramer's V": round(v, 3)
    })

association_table = pd.DataFrame(association_results)
association_table = tabulate(association_table, headers="keys", tablefmt="rounded_grid", floatfmt=".3f")
logToFile('\n### TEST CHI QUADRO E V-CRAMER', pathDebug)
logToFile(association_table, pathDebug)


### CONDIZIONE DI SALUTE VS DEPRESSIONE

health_table = pd.crosstab(
    df["Health_condition"], df["Depression"], margins=True
)

health_table["Tasso_depressione (%)"] = (
    health_table["Yes"] / health_table["All"] * 100
).round(2)

health_table = tabulate(
    health_table, headers="keys", tablefmt="rounded_grid"
)

logToFile("\n### CONDIZIONE DI SALUTE VS DEPRESSIONE", pathDebug)
logToFile(health_table, pathDebug)


### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE

rate = pd.crosstab(
    df["Relationship_status"], df["friend_circle"],
    values=(df["Depression"] == "Yes"), aggfunc="mean"
).mul(100)

n = pd.crosstab(df["Relationship_status"], df["friend_circle"])

labels = rate.round(1).astype(str) + "%\n(n=" + n.astype(str) + ")"

fig, ax = plt.subplots(figsize=(8, 6))

sns.heatmap(
    rate, annot=labels, fmt="", cmap="Oranges",
    cbar_kws={"label": "Tasso di depressione (%)"}, ax=ax
)

ax.set_title("Depression by Relationship Status and Friend Circle")
ax.set_xlabel("Friend Circle")
ax.set_ylabel("Relationship Status")

plt.tight_layout()
plt.savefig(pathPlots+"plot_Relationship_FriendCircle.png", format="png", bbox_inches="tight")


### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE ###

friend_results = []
relationship_groups = df["Relationship_status"].dropna().unique()

for group in relationship_groups:
    subset = df[df["Relationship_status"] == group]

    rate_yes = (subset.loc[subset["friend_circle"] == "Yes", "Depression"] == "Yes").mean() * 100
    rate_no = (subset.loc[subset["friend_circle"] == "No", "Depression"] == "Yes").mean() * 100

    table = pd.crosstab(subset["friend_circle"], subset["Depression"])
    p = chi2_contingency(table)[1] if table.shape[0] > 1 and table.shape[1] > 1 else np.nan

    friend_results.append({
        "Relationship_status": group,
        "Students": len(subset),
        "Rate (friends)": round(rate_yes, 1),
        "Rate (no friends)": round(rate_no, 1),
        "Difference": round(rate_no - rate_yes, 1),
        "P-value": p,
        "P-value (corrected)": min(p * len(relationship_groups), 1) if pd.notna(p) else np.nan
    })

friend_table = pd.DataFrame(friend_results).sort_values("Students", ascending=False)

for col in ["P-value", "P-value (corrected)"]:
    friend_table[col] = friend_table[col].map(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}" if pd.notna(p) else "N/A"
    )

friend_table = tabulate(friend_table, headers="keys", tablefmt="rounded_grid", showindex=False)

logToFile("\n### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE", pathDebug)
logToFile(friend_table, pathDebug)


# COSTI EXTRA-STUDIO E DEPRESSIONE #

df["Cost_quintile"] = pd.qcut(
    df["Cost_without_study"],
    q=5,
    labels=["Q1 (più basso)", "Q2", "Q3", "Q4", "Q5 (più alto)"]
)

cost_rates = (
    df.groupby("Cost_quintile", observed=True)["Depression"]
    .agg(
        Students="size",
        Rate=lambda x: (x == "Yes").mean() * 100
    )
    .reset_index()
)

### TABELLA DEI RISULTATI
cost_table = cost_rates.copy()
cost_table["Rate (%)"] = cost_table["Rate"].round(2)
cost_table = cost_table.drop(columns="Rate")

cost_table = tabulate(
    cost_table,
    headers="keys",
    tablefmt="rounded_grid",
    showindex=False
)

logToFile("\n### COSTI EXTRA-STUDIO E DEPRESSIONE", pathDebug)
logToFile(cost_table, pathDebug)

# Grafico
fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=cost_rates,
    x="Cost_quintile",
    y="Rate",
    color="tab:orange",
    ax=ax
)

### TASSO COMPLESSIVO DI DEPRESSIONE
overall_rate = (df["Depression"] == "Yes").mean() * 100

ax.axhline(
    overall_rate,
    color="gray",
    linestyle="--",
    label=f"Tasso complessivo ({overall_rate:.1f}%)"
)

ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
ax.set_title("Tasso di depressione per gruppo di costo extra-studio")
ax.set_xlabel("Costo extra-studio (quintili)")
ax.set_ylabel("Percentuale di studenti con depressione")
ax.legend()

plt.tight_layout()
plt.savefig(pathPlots+"plot_extra_studio_vs_depressione.png")
plt.close()


### TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE #

quintile_results = []

for q in df["Cost_quintile"].cat.categories:
    in_group = df["Cost_quintile"] == q

    rate_in = (df.loc[in_group, "Depression"] == "Yes").mean() * 100
    rate_out = (df.loc[~in_group, "Depression"] == "Yes").mean() * 100

    table = pd.crosstab(in_group, df["Depression"])
    p_value = chi2_contingency(table)[1]

    quintile_results.append({
        "Quintile": q,
        "Rate (quintile %)": round(rate_in, 1),
        "Rate (others %)": round(rate_out, 1),
        "P-value": p_value,
        "P-value (corrected)": min(p_value * 5, 1)
    })

quintile_table = pd.DataFrame(quintile_results)

for col in ["P-value", "P-value (corrected)"]:
    quintile_table[col] = quintile_table[col].map(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    )

quintile_table = tabulate(
    quintile_table,
    headers="keys",
    tablefmt="rounded_grid",
    showindex=False
)

logToFile("\n### TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE", pathDebug)
logToFile(quintile_table, pathDebug)


### COSTO E CLASSE SOCIALE #

median_cost = df.groupby("Family_background")["Cost_without_study"].median()
class_counts = pd.crosstab(df["Cost_quintile"], df["Family_background"])
class_rates = pd.crosstab(
    df["Cost_quintile"], df["Family_background"],
    values=df["Depression"].eq("Yes"), aggfunc="mean"
).mul(100)

class_rates = class_rates.round(1).astype(str) + "%"
class_rates = class_rates.mask(class_counts < 20, class_rates + "!")

for titolo, tabella in [
    ("Tasso di depressione (%)", class_rates),
    ("Tasso di depression N studenti", class_counts)
]:
    logToFile(f"\n### {titolo.upper()}", pathDebug)
    logToFile(tabulate(tabella, headers="keys", tablefmt="rounded_grid", stralign="right"), pathDebug)


### COSTI E DEPRESSIONE: CLASSE MEDIA E RICCA #

middle = df[df["Family_background"] == "Middle class"]
table = pd.crosstab(middle["Cost_quintile"], middle["Depression"])
chi2, p, dof, expected = chi2_contingency(table)
v = np.sqrt(chi2 / (table.to_numpy().sum() * min(table.shape[0]-1, table.shape[1]-1)))

logToFile(
    f"\n### CLASSE MEDIA: chi2={chi2:.2f}, dof={dof}, p-value={p:.5f}, V di Cramér={v:.3f}",
    pathDebug
)

rich = df[df["Family_background"] == "Rich class"]
q5 = rich["Cost_quintile"] == "Q5 (più alto)"

logToFile(
    f"### CLASSE RICCA: Q5={((rich.loc[q5, 'Depression'] == 'Yes').mean()*100):.1f}% (n={q5.sum()}), Q1-Q4={((rich.loc[~q5, 'Depression'] == 'Yes').mean()*100):.1f}% (n={(~q5).sum()})",
    pathDebug
)


### ASSOCIAZIONE VARIABILI CATEGORICHE E DEPRESSIONE #

association_results = []

for col in categorical_columns:
    table = pd.crosstab(df[col], df["Depression"])
    chi2, p, dof, expected = chi2_contingency(table)
    n, (r, k) = table.to_numpy().sum(), table.shape

    v = np.sqrt(chi2 / (n * min(r-1, k-1))) if min(r-1, k-1) > 0 else np.nan
    strength = "Negligible" if v < 0.1 else "Weak" if v < 0.3 else "Moderate" if v < 0.5 else "Strong"

    association_results.append({
        "Variable": col,
        "Chi-square": round(chi2, 2),
        "DoF": dof,
        "P-value": p,
        "Cramer's V": round(v, 3),
        "Strength": strength,
        "P-value corrected": min(p * len(categorical_columns), 1)
    })

association_table = pd.DataFrame(association_results).sort_values("Cramer's V", ascending=False)

for col in ["P-value", "P-value corrected"]:
    association_table[col] = association_table[col].map(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    )

association_table = tabulate(
    association_table, headers="keys", tablefmt="rounded_grid", showindex=False
)

logToFile("\n### ASSOCIAZIONE VARIABILI CATEGORICHE E DEPRESSIONE", pathDebug)
logToFile(association_table, pathDebug)


### DEPRESSIONE PER GENERE E STATO DI RELAZIONE #

gender_counts = pd.crosstab(df["Relationship_status"], df["Gender"])
gender_rates = pd.crosstab(
    df["Relationship_status"], df["Gender"],
    values=df["Depression"].eq("Yes"), aggfunc="mean"
).mul(100).round(1)

for titolo, tabella in [
    ("Tasso di depressione genere_vs_rel_status %", gender_rates),
    ("Tasso di depressione genere_vs_rel_status #N studenti", gender_counts)
]:
    logToFile(f"\n### {titolo.upper()}", pathDebug)
    logToFile(tabulate(tabella, headers="keys", tablefmt="rounded_grid"), pathDebug)


### HEATMAP V DI CRAMÉR 

cat_cols = categorical_columns + ["Depression"]
cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for a in cat_cols:
    for b in cat_cols:
        table = pd.crosstab(df[a], df[b])
        chi2 = chi2_contingency(table)[0]
        n = table.to_numpy().sum()
        denom = n * (min(table.shape) - 1)
        cramers_matrix.loc[a, b] = np.sqrt(chi2 / denom) if denom > 0 else np.nan

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cramers_matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("V di Cramér tra le variabili categoriche")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig(pathPlots+"plot_vCramer.png")
plt.close()


### ASSOCIAZIONE VARIABILI NUMERICHE CON DEPRESSIONE #

numeric_columns = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
results = []

n_no = (df["Depression"] == "No").sum()
n_yes = (df["Depression"] == "Yes").sum()

for col in numeric_columns:
    no_group = df.loc[df["Depression"] == "No", col].dropna()
    yes_group = df.loc[df["Depression"] == "Yes", col].dropna()

    statistic, p_value = mannwhitneyu(no_group, yes_group, alternative="two-sided")

    results.append({
        "Variable": col,
        "Median (No)": no_group.median(),
        "Median (Yes)": yes_group.median(),
        "U Statistic": statistic,
        "P-value": p_value,
        "Effect size (A)": statistic / (len(no_group) * len(yes_group))
    })

numeric_association_table = pd.DataFrame(results)

numeric_association_table["P-value corrected"] = (
    numeric_association_table["P-value"] * len(numeric_columns)
).clip(upper=1)

for col in ["P-value", "P-value corrected"]:
    numeric_association_table[col] = numeric_association_table[col].map(
        lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}"
    )

numeric_association_table[
    ["Median (No)", "Median (Yes)", "U Statistic", "Effect size (A)"]
] = numeric_association_table[
    ["Median (No)", "Median (Yes)", "U Statistic", "Effect size (A)"]
].round(2)

logToFile("\n### ASSOCIAZIONE VARIABILI NUMERICHE CON DEPRESSIONE", pathDebug)
logToFile(tabulate(numeric_association_table, headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)


### TASSO DI DEPRESSIONE PER QUINTILI DI CGPA #

df["CGPA_quintile"] = pd.qcut(
    df["CGPA"], q=5,
    labels=["C1 (più basso)", "C2", "C3", "C4", "C5 (più alto)"]
)

cgpa_rates = (
    df.groupby("CGPA_quintile", observed=True)
    .agg(
        CGPA_min=("CGPA", "min"),
        CGPA_max=("CGPA", "max"),
        Students=("Depression", "size"),
        Rate=("Depression", lambda x: (x == "Yes").mean() * 100)
    )
    .reset_index()
)

logToFile("\n### TASSO DI DEPRESSIONE PER QUINTILI DI CGPA", pathDebug)
logToFile(tabulate(cgpa_rates.round(2), headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(data=cgpa_rates, x="CGPA_quintile", y="Rate", color="tab:orange", ax=ax)

overall_rate = (df["Depression"] == "Yes").mean() * 100
ax.axhline(overall_rate, color="gray", linestyle="--",
           label=f"Tasso complessivo ({overall_rate:.1f}%)")

ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
ax.set_title("Tasso di depressione per gruppo di CGPA")
ax.set_xlabel("CGPA (quintili)")
ax.set_ylabel("Percentuale di studenti con depressione")
ax.legend()

plt.tight_layout()
plt.savefig(pathPlots+"plot_cgpa_quintili_depressione.png")
plt.close()


### TEST CHI-QUADRO E V DI CRAMÉR CGPA #

cgpa_table = pd.crosstab(df["CGPA_quintile"], df["Depression"])
chi2_cg, p_cg, dof_cg, _ = chi2_contingency(cgpa_table)
n_cg = cgpa_table.to_numpy().sum()
v_cg = np.sqrt(chi2_cg / (n_cg * min(cgpa_table.shape[0]-1, cgpa_table.shape[1]-1)))

cgpa_results = []

for q in df["CGPA_quintile"].cat.categories:
    mask = df["CGPA_quintile"] == q
    p = chi2_contingency(pd.crosstab(mask, df["Depression"]))[1]
    cgpa_results.append({
        "Quintile": q,
        "Rate (%)": (df.loc[mask, "Depression"] == "Yes").mean() * 100,
        "Rate others (%)": (df.loc[~mask, "Depression"] == "Yes").mean() * 100,
        "P-value": p,
        "P-value corr.": min(p * 5, 1),
        "V di Cramér": np.nan
    })

cgpa_results.extend([
    {"Quintile": "Test complessivo", "P-value": p_cg, "V di Cramér": np.nan},
    {"Quintile": "V di Cramér", "V di Cramér": v_cg}
])

cgpa_quintile_table = pd.DataFrame(cgpa_results)

for col in ["P-value", "P-value corr."]:
    cgpa_quintile_table[col] = cgpa_quintile_table[col].map(
        lambda p: "< 0.001" if pd.notna(p) and p < 0.001 else f"{p:.3f}" if pd.notna(p) else "-"
    )

logToFile("\n### TEST CHI-QUADRO E V DI CRAMÉR PER CGPA", pathDebug)
logToFile(tabulate(cgpa_quintile_table.fillna("-"), headers="keys",
                   tablefmt="rounded_grid", showindex=False), pathDebug)


# CORRELAZIONE DI SPEARMAN #

numeric_corr = df[numeric_columns].corr(method="spearman").round(2)

tabella_corr = numeric_corr.copy()
tabella_corr.insert(0, "Metodo", "Spearman")

logToFile("\n### CORRELAZIONE DI SPEARMAN", pathDebug)
logToFile(tabulate(
    tabella_corr, headers="keys", tablefmt="rounded_grid"
), pathDebug)

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    numeric_corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.8,
    annot_kws={"size": 12},
    ax=ax
)

ax.set_title("Correlazione di Spearman tra le variabili numeriche", pad=20)
ax.tick_params(axis="x", rotation=30, labelsize=11)
ax.tick_params(axis="y", rotation=0, labelsize=11)

plt.tight_layout()
plt.savefig(pathPlots+"plot_correlazione_spearman.png", bbox_inches="tight")
plt.close()


### PROFILI DIVERSI DI STUDENTI IN BASE ALLE VARIE CATEGORIE #

profile_columns = ["Relationship_status", "friend_circle", "Family_background", "Living_style"]

profile_summary = (
    df.groupby(profile_columns, observed=True)
    .agg(
        Total_Students=("Depression", "size"),
        Depression_Count=("Depression", lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

logToFile("\n### ANALISI PROFILI STUDENTI", pathDebug)
logToFile(f"Profili totali: {len(profile_summary)}", pathDebug)

profile_summary = profile_summary[profile_summary["Total_Students"] >= 20].copy()
logToFile(f"Profili con almeno 20 studenti: {len(profile_summary)}", pathDebug)

profile_summary["Depression_Rate"] = (
    profile_summary["Depression_Count"] / profile_summary["Total_Students"] * 100
)

ci_low, ci_high = proportion_confint(
    profile_summary["Depression_Count"],
    profile_summary["Total_Students"],
    method="wilson"
)

profile_summary["CI_low"] = ci_low * 100
profile_summary["CI_high"] = ci_high * 100

profile_summary[["Depression_Rate", "CI_low", "CI_high"]] = (
    profile_summary[["Depression_Rate", "CI_low", "CI_high"]].round(2)
)

profile_summary = profile_summary.sort_values("Depression_Rate", ascending=False)

logToFile(tabulate(profile_summary.head(10), headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)


### SEZIONE MODELLI

TARGET = "Depression"
num_cols = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
cat_cols = [
    col for col in categorical_columns 
    if col not in [TARGET, "Marital_status", "Health_condition"]
]

df_ready = df.copy()
df_tree = df_ready.copy()

# Rimuove eventuali colonne di clustering
cluster_cols = [col for col in df_tree.columns if col.startswith("cluster")]
df_tree = df_tree.drop(columns=cluster_cols, errors="ignore")

# Seleziona solo le colonne necessarie e rimuove righe incomplete
needed_cols = [TARGET] + num_cols + cat_cols
df_tree = df_tree[needed_cols].dropna().copy()

# Codifica del target
df_tree[TARGET] = (
    df_tree[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if df_tree[TARGET].isna().any():
    raise ValueError("Il target contiene valori diversi da Yes/No.")

df_tree[TARGET] = df_tree[TARGET].astype(int)

logToFile("\n### PREPARAZIONE DATASET MODELLI", pathDebug)
logToFile(f"Distribuzione target:\n{df_tree[TARGET].value_counts()}", pathDebug)
logToFile(f"Dimensioni finali dataset: {df_tree.shape}", pathDebug)


### SPLIT DEL DATASET IN TRAIN E TEST, STRATIFICANDO PER MANTENERE OMOGENEITA DEI DATI

X, y = df_tree[num_cols + cat_cols], df_tree[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=random_seed, stratify=y
)

logToFile("\n### SPLIT DATASET (Train/Test)", pathDebug)
logToFile(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}", pathDebug)
logToFile(f"\nDistribuzione target train (%):\n{y_train.value_counts(normalize=True).round(3)}", pathDebug)
logToFile(f"\nDistribuzione target test (%):\n{y_test.value_counts(normalize=True).round(3)}", pathDebug)


### Encoding delle variabili categoriche

X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=float)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=float).reindex(columns=X_train_encoded.columns, fill_value=0)

logToFile("\n### ENCODING FEATURE CATEGORICHE", pathDebug)
logToFile(f"Train encoded: {X_train_encoded.shape} | Test encoded: {X_test_encoded.shape}", pathDebug)
logToFile(f"Feature finali:\n{X_train_encoded.columns.tolist()}", pathDebug)


### Albero baseline

baseline_tree = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=20, class_weight="balanced", random_state=random_seed
).fit(X_train_encoded, y_train)

y_pred = baseline_tree.predict(X_test_encoded)
y_prob = baseline_tree.predict_proba(X_test_encoded)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

logToFile("\n### ALBERO BASELINE - RISULTATI TEST", pathDebug)
logToFile(f"Baseline Tree [Test] -> Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}", pathDebug)


### OTTIMIZZAZIONE DEI PARAMETRI GRID SEARCH

grid_params_path = "data/gridSearch.par"
run_gridsearch = True

if os.path.exists(grid_params_path):
    scelta = input(f"\nTrovato il file '{grid_params_path}'. Vuoi utilizzare i parametri salvati? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_gridsearch = False
        best_params = joblib.load(grid_params_path)
        best_tree = DecisionTreeClassifier(
            **best_params, class_weight="balanced", random_state=random_seed
        ).fit(X_train_encoded, y_train)
        logToFile("\n### GRIDSEARCH / PARAMETRI OTTIMALI", pathDebug)
        logToFile(f"Caricamento iperparametri salvati da '{grid_params_path}':\n{best_params}", pathDebug)

if run_gridsearch:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    param_grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 5, 10, 20, 30],
        "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02]
    }

    total_fits = np.prod([len(v) for v in param_grid.values()]) * cv.n_splits
    grid_search = GridSearchCV(
        DecisionTreeClassifier(class_weight="balanced", random_state=random_seed),
        param_grid, scoring="accuracy", cv=cv, n_jobs=-1
    )

    with tqdm_joblib(total=total_fits, desc="GridSearch Tree"):
        grid_search.fit(X_train_encoded, y_train)

    best_params = grid_search.best_params_
    os.makedirs("data", exist_ok=True)
    joblib.dump(best_params, grid_params_path)
    best_tree = grid_search.best_estimator_

    logToFile("\n### GRIDSEARCH / PARAMETRI OTTIMALI", pathDebug)
    logToFile(f"Migliori iperparametri: {best_params}", pathDebug)
    logToFile(f"Miglior CV score (Accuracy): {grid_search.best_score_:.3f}", pathDebug)


# Valutazione finale
y_pred = best_tree.predict(X_test_encoded)
y_prob = best_tree.predict_proba(X_test_encoded)[:, 1]

acc, prec, rec, f1, auc = (
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred),
    recall_score(y_test, y_pred),
    f1_score(y_test, y_pred),
    roc_auc_score(y_test, y_prob)
)

cm_table = tabulate(confusion_matrix(y_test, y_pred), headers=["Pred No (0)", "Pred Yes (1)"], tablefmt="rounded_grid")

logToFile("\n### MODELLO FINALE (BEST TREE) - VALUTAZIONE TEST SET", pathDebug)
logToFile(f"Best Tree [Test] -> Acc: {acc:.3f} | Prec: {prec:.3f} | Rec: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}", pathDebug)
logToFile(f"\nConfusion matrix:\n{cm_table}", pathDebug)
logToFile(f"\nClassification report:\n{classification_report(y_test, y_pred)}", pathDebug)

### PLOT ROC CURVE

from sklearn.metrics import RocCurveDisplay

RocCurveDisplay.from_predictions(y_test, y_prob, name="Decision Tree")
plt.title("ROC curve — Decision Tree")
plt.tight_layout()

plt.savefig(pathPlots + "plot_roc_curve.png")
plt.savefig(pathPlots + "plot_roc_curve.svg")
plt.close()

### PLOT ROC CURVE

plt.figure(figsize=(20, 10))
plot_tree(
    best_tree,
    feature_names=X_train_encoded.columns,
    class_names=["No Depression", "Depression"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9
)
plt.title("Albero di classificazione — primi 3 livelli")
plt.tight_layout()

plt.savefig(pathPlots + "plot_decision_tree.png")
plt.savefig(pathPlots + "plot_decision_tree.svg")
plt.close()

### PLOT VARIABILI PIU IMPORTANTI

importances = (
    pd.DataFrame({
        "Feature": X_train_encoded.columns,
        "Importance": best_tree.feature_importances_
    })
    .sort_values("Importance", ascending=False)
    .head(15)
)

logToFile("\n### FEATURE IMPORTANCE - TOP 15", pathDebug)
logToFile(tabulate(importances, headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importances, x="Importance", y="Feature", palette="Oranges_r", ax=ax)
ax.set_title("Feature Importance — Decision Tree")
plt.tight_layout()

plt.savefig(pathPlots + "plot_feature_importance.png")
plt.savefig(pathPlots + "plot_feature_importance.svg")
plt.close()


from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay

### SEZIONE RANDOM FOREST CON SOGLIA DI DECISIONE E OOB ANALYSIS

TARGET = "Depression"
num_cols = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
cat_cols = [
    "Gender", "Family_background", "Relationship_status",
    "Marital_status", "Living_style", "friend_circle", "Home_division"
]
excluded_cols = ["Health_condition"]

logToFile("\n### CONFIGURAZIONE RANDOM FOREST", pathDebug)
logToFile(f"Target: {TARGET}\nNumeriche: {num_cols}\nCategoriche: {cat_cols}\nEscluse: {excluded_cols}", pathDebug)

# Preparazione Dataset Random Forest
df_rf = df_ready.copy()
cluster_cols = [col for col in df_rf.columns if col.startswith("cluster")]
df_rf = df_rf.drop(columns=cluster_cols, errors="ignore")

needed_cols = [TARGET] + num_cols + cat_cols
df_rf = df_rf[needed_cols].dropna(subset=needed_cols).copy()

df_rf[TARGET] = (
    df_rf[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

if df_rf[TARGET].isna().any():
    raise ValueError("Il target contiene valori diversi da Yes/No.")

df_rf[TARGET] = df_rf[TARGET].astype(int)

logToFile("\n### PREPARAZIONE DATASET RANDOM FOREST", pathDebug)
logToFile(f"Distribuzione del target:\n{df_rf[TARGET].value_counts(dropna=False)}", pathDebug)
logToFile(f"Dimensioni dataset: {df_rf.shape}", pathDebug)

# Train/Test Split
X = df_rf[num_cols + cat_cols]
y = df_rf[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=random_seed, stratify=y
)

logToFile("\n### SPLIT DATASET RANDOM FOREST (Train/Test)", pathDebug)
logToFile(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}", pathDebug)
logToFile(f"Distribuzione target train (%):\n{y_train.value_counts(normalize=True).round(3)}", pathDebug)
logToFile(f"Distribuzione target test (%):\n{y_test.value_counts(normalize=True).round(3)}", pathDebug)

# Preprocessing & Pipeline Base
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(
    handle_unknown="ignore", drop="first", sparse_output=False
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ],
    remainder="drop"
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300, random_state=random_seed, n_jobs=-1, class_weight="balanced"
        ))
    ]
)

# Hyperparameter Tuning con RandomizedSearchCV
rf_params_path = "data/randomSearchRandomForest.par"
run_randomsearch = True

if os.path.exists(rf_params_path):
    scelta = input(f"\nTrovato il file '{rf_params_path}'. Vuoi utilizzare i parametri salvati? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_randomsearch = False
        saved_params = joblib.load(rf_params_path)
        best_params_extracted = saved_params
        logToFile("\n### RANDOMIZED SEARCH RF / PARAMETRI OTTIMALI CARICATI", pathDebug)
        logToFile(f"Caricamento iperparametri salvati da '{rf_params_path}':\n{saved_params}", pathDebug)

if run_randomsearch:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    param_distributions = {
        "classifier__max_depth": [4, 6, 8, 10, 15, None],
        "classifier__min_samples_split": randint(2, 21),
        "classifier__min_samples_leaf": randint(1, 21),
        "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "classifier__class_weight": ["balanced", "balanced_subsample"]
    }

    n_iter = 60
    total_fits = n_iter * cv.n_splits

    random_search = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        random_state=random_seed,
        return_train_score=True,
        refit=True
    )

    with tqdm_joblib(total=total_fits, desc="RandomSearch RF"):
        random_search.fit(X_train, y_train)

    os.makedirs("data", exist_ok=True)
    joblib.dump(random_search.best_params_, rf_params_path)

    best_params_extracted = {
        key.replace("classifier__", ""): value
        for key, value in random_search.best_params_.items()
    }

    logToFile("\n### RANDOMIZED SEARCH RF / PARAMETRI OTTIMALI", pathDebug)
    logToFile(f"Migliori iperparametri: {random_search.best_params_}", pathDebug)
    logToFile(f"Miglior F1 score in CV: {random_search.best_score_:.3f}", pathDebug)

    cv_results = pd.DataFrame(random_search.cv_results_)
    top_results = (
        cv_results.sort_values("rank_test_score")
        .loc[:, ["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "params"]]
        .head(10)
    )
    logToFile(f"\nTop 10 combinazioni CV:\n{tabulate(top_results, headers='keys', tablefmt='rounded_grid', showindex=False)}", pathDebug)

# Pulizia chiavi iperparametri per modello standalone (se ricaricati)
best_params_clean = {
    key.replace("classifier__", ""): value
    for key, value in best_params_extracted.items()
}

# OOB Curve Analysis
X_train_transformed = preprocessor.fit_transform(X_train)

oob_rf = RandomForestClassifier(
    **best_params_clean,
    n_estimators=50,
    oob_score=True,
    bootstrap=True,
    warm_start=True,
    random_state=random_seed,
    n_jobs=-1
)

n_trees = []
oob_errors = []
oob_accuracies = []
oob_auc_scores = []

for n in range(50, 601, 25):
    oob_rf.set_params(n_estimators=n)
    oob_rf.fit(X_train_transformed, y_train)
    oob_prob = oob_rf.oob_decision_function_[:, 1]

    n_trees.append(n)
    oob_accuracies.append(oob_rf.oob_score_)
    oob_errors.append(1 - oob_rf.oob_score_)
    oob_auc_scores.append(roc_auc_score(y_train, oob_prob))

oob_results = pd.DataFrame({
    "n_estimators": n_trees,
    "OOB_accuracy": oob_accuracies,
    "OOB_error": oob_errors,
    "OOB_ROC_AUC": oob_auc_scores
})

best_n = int(oob_results.loc[oob_results["OOB_ROC_AUC"].idxmax(), "n_estimators"])

logToFile("\n### ANALISI OOB PER NUMERO DI ALBERI", pathDebug)
logToFile(f"Numero di alberi con OOB ROC-AUC massimo: {best_n}", pathDebug)
logToFile(f"OOB ROC-AUC massimo: {max(oob_auc_scores):.4f}", pathDebug)
logToFile(f"OOB error corrispondente: {min(oob_errors):.4f}", pathDebug)

# Plot OOB Error vs OOB ROC-AUC
fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(oob_results["n_estimators"], oob_results["OOB_error"], marker="o", color="tab:red", label="OOB error")
ax1.set_xlabel("Numero di alberi")
ax1.set_ylabel("OOB error", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:red")

ax2 = ax1.twinx()
ax2.plot(oob_results["n_estimators"], oob_results["OOB_ROC_AUC"], marker="o", color="tab:green", label="OOB ROC-AUC")
ax2.set_ylabel("OOB ROC-AUC", color="tab:green")
ax2.tick_params(axis="y", labelcolor="tab:green")

ax1.axvline(best_n, color="black", linestyle="--", label=f"Scelta: {best_n} alberi")
fig.suptitle("Andamento OOB in funzione del numero di alberi")
fig.tight_layout()

plt.savefig(pathPlots + "plot_rf_oob_curve.png")
plt.savefig(pathPlots + "plot_rf_oob_curve.svg")
plt.close()

# Modello Finale Random Forest
final_n_estimators = max(best_n, 300)

final_rf = RandomForestClassifier(
    **best_params_clean,
    n_estimators=final_n_estimators,
    oob_score=True,
    bootstrap=True,
    random_state=random_seed,
    n_jobs=-1
)

final_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", final_rf)
    ]
)

final_pipeline.fit(X_train, y_train)
rf_final = final_pipeline.named_steps["classifier"]

oob_prob_train = rf_final.oob_decision_function_[:, 1]
logToFile("\n### VALUTAZIONE OOB MODELLO FINALE RF", pathDebug)
logToFile(f"Numero finale di alberi: {final_n_estimators}", pathDebug)
logToFile(f"OOB accuracy: {rf_final.oob_score_:.4f} | OOB error: {1 - rf_final.oob_score_:.4f}", pathDebug)
logToFile(f"OOB ROC-AUC: {roc_auc_score(y_train, oob_prob_train):.4f}", pathDebug)

# Valutazione Test Set — Soglia Standard (0.50)
y_prob = final_pipeline.predict_proba(X_test)[:, 1]
y_pred_default = (y_prob >= 0.50).astype(int)

cm_default = tabulate(confusion_matrix(y_test, y_pred_default), headers=["Pred No (0)", "Pred Yes (1)"], tablefmt="rounded_grid")

logToFile("\n### RANDOM FOREST - TEST SET (SOGLIA DEFAULT 0.50)", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_default):.3f} | Prec: {precision_score(y_test, y_pred_default):.3f} | Rec: {recall_score(y_test, y_pred_default):.3f} | F1: {f1_score(y_test, y_pred_default):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}", pathDebug)
logToFile(f"\nConfusion Matrix:\n{cm_default}", pathDebug)
logToFile(f"\nClassification Report:\n{classification_report(y_test, y_pred_default)}", pathDebug)

# Ottimizzazione della Soglia di Probabilità (F1 Maximization)
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

threshold_table = pd.DataFrame({
    "threshold": thresholds,
    "precision": precisions[:-1],
    "recall": recalls[:-1]
})

threshold_table["f1"] = (
    2 * threshold_table["precision"] * threshold_table["recall"]
    / (threshold_table["precision"] + threshold_table["recall"] + 1e-12)
)

best_idx = threshold_table["f1"].idxmax()
best_threshold = threshold_table.loc[best_idx, "threshold"]

logToFile("\n### OTTIMIZZAZIONE SOGLIA DI DECISIONE", pathDebug)
logToFile(f"Soglia ottimale per F1: {best_threshold:.4f}", pathDebug)
logToFile(tabulate(threshold_table.loc[[best_idx]].round(4), headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

# Plot Precision, Recall e F1 vs Soglia
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(threshold_table["threshold"], threshold_table["precision"], label="Precision", color="tab:blue")
ax.plot(threshold_table["threshold"], threshold_table["recall"], label="Recall", color="tab:orange")
ax.plot(threshold_table["threshold"], threshold_table["f1"], label="F1-score", color="tab:green", linewidth=2)
ax.axvline(best_threshold, color="black", linestyle="--", label=f"Soglia ottimale = {best_threshold:.3f}")
ax.set_xlabel("Soglia di probabilità")
ax.set_ylabel("Metrica")
ax.set_ylim(0, 1.05)
ax.set_title("Precision, Recall e F1 in funzione della soglia")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(pathPlots + "plot_rf_threshold_tuning.png")
plt.savefig(pathPlots + "plot_rf_threshold_tuning.svg")
plt.close()

# Valutazione Test Set — Soglia Ottimizzata
y_pred_opt = (y_prob >= best_threshold).astype(int)
cm_opt = tabulate(confusion_matrix(y_test, y_pred_opt), headers=["Pred No (0)", "Pred Yes (1)"], tablefmt="rounded_grid")

logToFile("\n### RANDOM FOREST - TEST SET (SOGLIA OTTIMIZZATA)", pathDebug)
logToFile(f"Soglia applicata: {best_threshold:.4f}", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_opt):.3f} | Prec: {precision_score(y_test, y_pred_opt):.3f} | Rec: {recall_score(y_test, y_pred_opt):.3f} | F1: {f1_score(y_test, y_pred_opt):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}", pathDebug)
logToFile(f"\nConfusion Matrix:\n{cm_opt}", pathDebug)
logToFile(f"\nClassification Report:\n{classification_report(y_test, y_pred_opt)}", pathDebug)

# Plot Curva Precision-Recall
fig, ax = plt.subplots(figsize=(8, 7))
PrecisionRecallDisplay.from_predictions(
    y_test, y_prob, name="Random Forest", ax=ax, plot_chance_level=True
)
ax.scatter(
    threshold_table.loc[best_idx, "recall"],
    threshold_table.loc[best_idx, "precision"],
    color="tab:red", s=80, zorder=5,
    label=f"Soglia ottimale F1 = {best_threshold:.3f}"
)
ax.set_title("Curva Precision-Recall — Random Forest")
ax.legend(loc="lower left")
ax.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(pathPlots + "plot_rf_precision_recall_curve.png")
plt.savefig(pathPlots + "plot_rf_precision_recall_curve.svg")
plt.close()

# Plot Curva ROC
fig, ax = plt.subplots(figsize=(8, 7))
RocCurveDisplay.from_predictions(y_test, y_prob, name="Random Forest", ax=ax)
ax.set_title("Curva ROC — Random Forest")
plt.tight_layout()

plt.savefig(pathPlots + "plot_rf_roc_curve.png")
plt.savefig(pathPlots + "plot_rf_roc_curve.svg")
plt.close()

# Feature Importance Random Forest
feature_names = final_pipeline.named_steps["preprocessor"].get_feature_names_out()

importances_rf = pd.DataFrame({
    "Feature": feature_names,
    "Importance": rf_final.feature_importances_
}).sort_values(by="Importance", ascending=False).head(15)

logToFile("\n### FEATURE IMPORTANCE RANDOM FOREST (TOP 15)", pathDebug)
logToFile(tabulate(importances_rf, headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(data=importances_rf, x="Importance", y="Feature", palette="Blues_r", ax=ax)
ax.set_title("Feature importance — Random Forest")
plt.tight_layout()

plt.savefig(pathPlots + "plot_rf_feature_importance.png")
plt.savefig(pathPlots + "plot_rf_feature_importance.svg")
plt.close()