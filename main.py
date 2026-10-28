### Import delle librerie

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
from tabulate import tabulate
warnings.filterwarnings("ignore")


from scipy.stats import chi2_contingency


plt.style.use("fivethirtyeight")

def logToFile(data, output):
    '''
    
    '''
    print(data)
    if output != None:
        with open(output, 'a', encoding='utf-8') as file:
            file.write(data + '\n')

### Pulizia file di log
if os.path.exists("debug.txt"):
    os.remove('debug.txt')

### Import dei file

path_file = r'Student_Depression_Dataset.csv'
df = pd.read_csv(path_file)


### Analisi missing values
colMissing = df.isna().sum().to_frame(name='Valori mancanti')
colMissing = tabulate(colMissing, headers="keys", tablefmt='rounded_grid')

logToFile('\n### ANALISI DATI MANCANTI', 'debug.txt')
logToFile(colMissing, 'debug.txt')


stats = df.describe()
stats = tabulate(stats, headers="keys", tablefmt="rounded_grid", floatfmt=".2f")

logToFile('\n### Analisi statistica colonne numeriche', 'debug.txt')
logToFile(stats, 'debug.txt')


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

plt.savefig("plot_AgeDistribution_Gender.svg", format="svg", bbox_inches="tight")


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
plt.savefig("plot_DepressionStatus.svg", format="svg", bbox_inches="tight")


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
    plt.savefig("plot_"+column+"_vs_Depression.svg", format="svg", bbox_inches="tight")

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
logToFile('\n### STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE','debug.txt')
logToFile(depression_rates, 'debug.txt')


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
logToFile('\n### TEST CHI QUADRO E V-CRAMER','debug.txt')
logToFile(association_table, 'debug.txt')


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

logToFile("\n### CONDIZIONE DI SALUTE VS DEPRESSIONE", "debug.txt")
logToFile(health_table, "debug.txt")


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

logToFile("\n### CONDIZIONE DI SALUTE VS DEPRESSIONE", "debug.txt")
logToFile(health_table, "debug.txt")


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
plt.savefig("plot_Relationship_FriendCircle.svg", format="svg", bbox_inches="tight")


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

logToFile("\n### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE", "debug.txt")
logToFile(friend_table, "debug.txt")


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

# Tabella dei risultati
cost_table = cost_rates.copy()
cost_table["Rate (%)"] = cost_table["Rate"].round(2)
cost_table = cost_table.drop(columns="Rate")

cost_table = tabulate(
    cost_table,
    headers="keys",
    tablefmt="rounded_grid",
    showindex=False
)

logToFile("\n### COSTI EXTRA-STUDIO E DEPRESSIONE", "debug.txt")
logToFile(cost_table, "debug.txt")

# Grafico
fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=cost_rates,
    x="Cost_quintile",
    y="Rate",
    color="tab:orange",
    ax=ax
)

# Tasso complessivo di depressione
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
plt.savefig("plot_extra_studio_vs_depressione.svg")
plt.close()


# TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE #

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

logToFile("\n### TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE", "debug.txt")
logToFile(quintile_table, "debug.txt")


# COSTO E CLASSE SOCIALE #

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
    logToFile(f"\n### {titolo.upper()}", "debug.txt")
    logToFile(tabulate(tabella, headers="keys", tablefmt="rounded_grid", stralign="right"), "debug.txt")



