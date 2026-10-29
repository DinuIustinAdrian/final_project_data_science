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

plt.style.use("fivethirtyeight")

def logToFile(data, output):
    print(data)
    if output != None:
        with open(output, 'a', encoding='utf-8') as file:
            file.write(data + '\n')

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
print("\nRELATIONSHIP STATUS VS MARITAL STATUS:")
print(pd.crosstab(df["Relationship_status"], df["Marital_status"]))


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

# Rimozione depression perche ovvia



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
logToFile('\n### STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE',pathDebug)
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
logToFile('\n### TEST CHI QUADRO E V-CRAMER',pathDebug)
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

logToFile(tabulate(cgpa_rates.round(2),headers="keys",tablefmt="rounded_grid",showindex=False), pathDebug)

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


### TEST CHI-QUADRO E V DI CRAMÉR #

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

logToFile(tabulate(cgpa_quintile_table.fillna("-"), headers="keys",
                   tablefmt="rounded_grid", showindex=False), pathDebug)



# CORRELAZIONE DI SPEARMAN #

numeric_corr = df[numeric_columns].corr(method="spearman").round(2)

tabella_corr = numeric_corr.copy()
tabella_corr.insert(0, "Metodo", "Spearman")

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

logToFile(tabulate(profile_summary.head(10),headers="keys",tablefmt="rounded_grid",showindex=False), pathDebug)



### SEZIONE MODELLI


TARGET = "Depression"
num_cols = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
cat_cols = [col for col in categorical_columns if col != TARGET]

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

