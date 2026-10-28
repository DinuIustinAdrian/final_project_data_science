import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency, mannwhitneyu
from statsmodels.stats.proportion import proportion_confint
from tabulate import tabulate

warnings.filterwarnings("ignore")
plt.style.use("fivethirtyeight")


# FUNZIONI GENERALI

def logToFile(data, output="debug.txt"):
    print(data)

    if output is not None:
        with open(output, "a", encoding="utf-8") as file:
            file.write(str(data) + "\n")


def stampaTabella(tabella, titolo, output="debug.txt", showindex=False):
    tabella = tabulate(
        tabella,
        headers="keys",
        tablefmt="rounded_grid",
        showindex=showindex
    )
    logToFile("\n### " + titolo, output)
    logToFile(tabella, output)


def salvaGrafico(nome_file):
    plt.tight_layout()
    plt.savefig(nome_file, format="svg", bbox_inches="tight")
    plt.close()


def formattaPvalue(valore):
    if pd.isna(valore):
        return "N/A"
    if valore < 0.001:
        return "< 0.001"
    return f"{valore:.3f}"


def calcolaCramersV(tabella):
    chi2 = chi2_contingency(tabella)[0]
    n = tabella.to_numpy().sum()
    righe, colonne = tabella.shape
    denominatore = n * min(righe - 1, colonne - 1)

    if denominatore <= 0:
        return np.nan

    return np.sqrt(chi2 / denominatore)


def calcolaTasso(depressione):
    return (depressione == "Yes").mean() * 100


def creaTabellaAssociazione(colonna, target="Depression"):
    tabella = pd.crosstab(df[colonna], df[target])
    chi2, p_value, dof, expected = chi2_contingency(tabella)

    return {
        "Variable": colonna,
        "Chi-square": round(chi2, 2),
        "DoF": dof,
        "P-value": p_value,
        "Cramer's V": round(calcolaCramersV(tabella), 3)
    }


def plotTassoQuintili(risultati, colonna_x, titolo, etichetta_x, nome_file):
    fig, ax = plt.subplots(figsize=(9, 5))

    sns.barplot(
        data=risultati,
        x=colonna_x,
        y="Rate",
        color="tab:orange",
        ax=ax
    )

    tasso_totale = calcolaTasso(df["Depression"])
    ax.axhline(
        tasso_totale,
        color="gray",
        linestyle="--",
        label=f"Tasso complessivo ({tasso_totale:.1f}%)"
    )

    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    ax.set_title(titolo)
    ax.set_xlabel(etichetta_x)
    ax.set_ylabel("Percentuale di studenti con depressione")
    ax.legend()

    salvaGrafico(nome_file)


# PULIZIA E IMPORT DEI DATI

if os.path.exists("debug.txt"):
    os.remove("debug.txt")

path_file = "Student_Depression_Dataset.csv"
df = pd.read_csv(path_file)


# ANALISI DEI VALORI MANCANTI E STATISTICHE DESCRITTIVE

missing = df.isna().sum().to_frame(name="Valori mancanti")
stampaTabella(missing, "ANALISI DATI MANCANTI")

stampaTabella(
    df.describe().round(2),
    "ANALISI STATISTICA COLONNE NUMERICHE"
)

logToFile("\n### RELATIONSHIP STATUS VS MARITAL STATUS")
logToFile(pd.crosstab(df["Relationship_status"], df["Marital_status"]))


# DISTRIBUZIONE DELL'ETA PER GENERE

fig, ax1 = plt.subplots(figsize=(10, 6))

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
salvaGrafico("plot_AgeDistribution_Gender.svg")


# DISTRIBUZIONE DELLO STATO DI DEPRESSIONE

fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=df, x="Depression", ax=ax)

for container in ax.containers:
    etichette = []

    for barra in container:
        altezza = barra.get_height()
        percentuale = altezza / len(df) * 100
        etichette.append(f"{int(altezza)}\n({percentuale:.1f}%)")

    ax.bar_label(container, labels=etichette, padding=3)

ax.set_title("Reported Depression Status Among Students", fontsize=14, pad=12)
ax.set_xlabel("Depression Status")
ax.set_ylabel("Number of Students")
ax.set_ylim(top=ax.get_ylim()[1] * 1.15)
salvaGrafico("plot_DepressionStatus.svg")


# DEPRESSIONE IN BASE ALLE CATEGORIE

def plot_depression_comparison(column, title=None):
    titolo = title or column.replace("_", " ").title()

    counts = pd.crosstab(df[column], df["Depression"]).reindex(
        columns=["No", "Yes"], fill_value=0
    )
    percentuali = counts.div(counts.sum(axis=1), axis=0) * 100
    ordine = counts.sum(axis=1).sort_values(ascending=False).index

    counts = counts.loc[ordine]
    percentuali = percentuali.loc[ordine]

    ax = counts.plot(
        kind="bar",
        stacked=True,
        figsize=(max(8, len(ordine) * 1.2), 6),
        color=["tab:blue", "tab:orange"]
    )

    for i, container in enumerate(ax.containers):
        etichette = [
            f"{n}\n({p:.1f}%)" if n > 0 else ""
            for n, p in zip(counts.iloc[:, i], percentuali.iloc[:, i])
        ]
        ax.bar_label(
            container,
            labels=etichette,
            label_type="center",
            fontsize=12
        )

    ax.set(
        title=f"{titolo} and Depression",
        xlabel=titolo,
        ylabel="Number of Students"
    )
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Depression")
    ax.grid(axis="y", alpha=0.8)

    salvaGrafico("plot_" + column + "_vs_Depression.svg")


colonne_grafici = [
    "Family_background",
    "Relationship_status",
    "Marital_status",
    "Living_style",
    "Health_condition",
    "friend_circle"
]

for colonna in colonne_grafici:
    plot_depression_comparison(colonna)


# ELENCO DELLE VARIABILI CATEGORICHE

categorical_columns = [
    col for col in df.select_dtypes(include=["object", "category", "string"]).columns
    if col != "Depression"
]


# TASSO DI DEPRESSIONE PER SOTTOCATEGORIA

summary_list = []

for colonna in categorical_columns:
    summary = (
        df.groupby(colonna, observed=True)["Depression"]
        .apply(calcolaTasso)
        .reset_index(name="Depression_Rate")
    )

    summary["Variable"] = colonna
    summary = summary.rename(columns={colonna: "Category"})
    summary_list.append(summary)

depression_rates = pd.concat(summary_list, ignore_index=True)
depression_rates = depression_rates[
    ["Variable", "Category", "Depression_Rate"]
].sort_values("Depression_Rate", ascending=False)
depression_rates["Depression_Rate"] = depression_rates["Depression_Rate"].round(2)

stampaTabella(
    depression_rates,
    "STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE"
)


# TEST CHI-QUADRATO E V DI CRAMER

association_results = []

for colonna in categorical_columns:
    risultato = creaTabellaAssociazione(colonna)
    risultato["P-value"] = formattaPvalue(risultato["P-value"])
    association_results.append(risultato)

stampaTabella(
    pd.DataFrame(association_results),
    "TEST CHI QUADRO E V-CRAMER"
)


# CONDIZIONE DI SALUTE VS DEPRESSIONE

health_table = pd.crosstab(
    df["Health_condition"],
    df["Depression"],
    margins=True
)

health_table["Tasso_depressione (%)"] = (
    health_table["Yes"] / health_table["All"] * 100
).round(2)

stampaTabella(health_table, "CONDIZIONE DI SALUTE VS DEPRESSIONE")


# STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE

rate = pd.crosstab(
    df["Relationship_status"],
    df["friend_circle"],
    values=df["Depression"].eq("Yes"),
    aggfunc="mean"
) * 100

counts = pd.crosstab(df["Relationship_status"], df["friend_circle"])
labels = rate.round(1).astype(str) + "%\n(n=" + counts.astype(str) + ")"

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    rate,
    annot=labels,
    fmt="",
    cmap="Oranges",
    cbar_kws={"label": "Tasso di depressione (%)"},
    ax=ax
)

ax.set_title("Depression by Relationship Status and Friend Circle")
ax.set_xlabel("Friend Circle")
ax.set_ylabel("Relationship Status")
salvaGrafico("plot_Relationship_FriendCircle.svg")


friend_results = []
relationship_groups = df["Relationship_status"].dropna().unique()

for gruppo in relationship_groups:
    subset = df[df["Relationship_status"] == gruppo]

    rate_yes = calcolaTasso(
        subset.loc[subset["friend_circle"] == "Yes", "Depression"]
    )
    rate_no = calcolaTasso(
        subset.loc[subset["friend_circle"] == "No", "Depression"]
    )

    tabella = pd.crosstab(subset["friend_circle"], subset["Depression"])

    if tabella.shape[0] > 1 and tabella.shape[1] > 1:
        p_value = chi2_contingency(tabella)[1]
    else:
        p_value = np.nan

    friend_results.append({
        "Relationship_status": gruppo,
        "Students": len(subset),
        "Rate (friends)": round(rate_yes, 1),
        "Rate (no friends)": round(rate_no, 1),
        "Difference": round(rate_no - rate_yes, 1),
        "P-value": p_value,
        "P-value (corrected)": min(p_value * len(relationship_groups), 1)
        if pd.notna(p_value) else np.nan
    })

friend_table = pd.DataFrame(friend_results).sort_values(
    "Students", ascending=False
)

for colonna in ["P-value", "P-value (corrected)"]:
    friend_table[colonna] = friend_table[colonna].map(formattaPvalue)

stampaTabella(
    friend_table,
    "STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE"
)


# COSTI EXTRA-STUDIO E DEPRESSIONE

df["Cost_quintile"] = pd.qcut(
    df["Cost_without_study"],
    q=5,
    labels=["Q1 (più basso)", "Q2", "Q3", "Q4", "Q5 (più alto)"]
)

cost_rates = (
    df.groupby("Cost_quintile", observed=True)
    .agg(
        Students=("Depression", "size"),
        Rate=("Depression", calcolaTasso)
    )
    .reset_index()
)

cost_table = cost_rates.copy()
cost_table["Rate (%)"] = cost_table["Rate"].round(2)
cost_table = cost_table.drop(columns="Rate")
stampaTabella(cost_table, "COSTI EXTRA-STUDIO E DEPRESSIONE")

plotTassoQuintili(
    cost_rates,
    "Cost_quintile",
    "Tasso di depressione per gruppo di costo extra-studio",
    "Costo extra-studio (quintili)",
    "plot_extra_studio_vs_depressione.svg"
)


# TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE

quintile_results = []

for quintile in df["Cost_quintile"].cat.categories:
    nel_gruppo = df["Cost_quintile"] == quintile

    rate_in = calcolaTasso(df.loc[nel_gruppo, "Depression"])
    rate_out = calcolaTasso(df.loc[~nel_gruppo, "Depression"])
    tabella = pd.crosstab(nel_gruppo, df["Depression"])
    p_value = chi2_contingency(tabella)[1]

    quintile_results.append({
        "Quintile": quintile,
        "Rate (quintile %)": round(rate_in, 1),
        "Rate (others %)": round(rate_out, 1),
        "P-value": formattaPvalue(p_value),
        "P-value (corrected)": formattaPvalue(min(p_value * 5, 1))
    })

stampaTabella(
    pd.DataFrame(quintile_results),
    "TEST STATISTICO: QUINTILI DI COSTO E DEPRESSIONE"
)


# COSTO E CLASSE SOCIALE

class_counts = pd.crosstab(df["Cost_quintile"], df["Family_background"])
class_rates = pd.crosstab(
    df["Cost_quintile"],
    df["Family_background"],
    values=df["Depression"].eq("Yes"),
    aggfunc="mean"
) * 100

class_rates = class_rates.round(1).astype(str) + "%"
class_rates = class_rates.mask(class_counts < 20, class_rates + "!")

stampaTabella(class_rates, "TASSO DI DEPRESSIONE (%)", showindex=True)
stampaTabella(class_counts, "NUMERO DI STUDENTI", showindex=True)


# COSTI E DEPRESSIONE: CLASSE MEDIA E RICCA

middle = df[df["Family_background"] == "Middle class"]
table = pd.crosstab(middle["Cost_quintile"], middle["Depression"])
chi2, p_value, dof, expected = chi2_contingency(table)
v = calcolaCramersV(table)

logToFile(
    f"\n### CLASSE MEDIA: chi2={chi2:.2f}, dof={dof}, "
    f"p-value={p_value:.5f}, V di Cramér={v:.3f}"
)

rich = df[df["Family_background"] == "Rich class"]
q5 = rich["Cost_quintile"] == "Q5 (più alto)"

logToFile(
    f"### CLASSE RICCA: Q5={calcolaTasso(rich.loc[q5, 'Depression']):.1f}% "
    f"(n={q5.sum()}), Q1-Q4="
    f"{calcolaTasso(rich.loc[~q5, 'Depression']):.1f}% "
    f"(n={(~q5).sum()})"
)


# ASSOCIAZIONE VARIABILI CATEGORICHE E DEPRESSIONE

association_results = []

for colonna in categorical_columns:
    risultato = creaTabellaAssociazione(colonna)
    v = risultato["Cramer's V"]

    if pd.isna(v):
        forza = "N/A"
    elif v < 0.1:
        forza = "Negligible"
    elif v < 0.3:
        forza = "Weak"
    elif v < 0.5:
        forza = "Moderate"
    else:
        forza = "Strong"

    risultato["Strength"] = forza
    risultato["P-value corrected"] = min(
        risultato["P-value"] * len(categorical_columns), 1
    )
    association_results.append(risultato)

association_table = pd.DataFrame(association_results).sort_values(
    "Cramer's V", ascending=False
)

for colonna in ["P-value", "P-value corrected"]:
    association_table[colonna] = association_table[colonna].map(formattaPvalue)

stampaTabella(
    association_table,
    "ASSOCIAZIONE VARIABILI CATEGORICHE E DEPRESSIONE"
)


# DEPRESSIONE PER GENERE E STATO DI RELAZIONE

gender_counts = pd.crosstab(df["Relationship_status"], df["Gender"])
gender_rates = pd.crosstab(
    df["Relationship_status"],
    df["Gender"],
    values=df["Depression"].eq("Yes"),
    aggfunc="mean"
) * 100

stampaTabella(gender_rates.round(1), "TASSO DI DEPRESSIONE GENERE VS RELATIONSHIP STATUS")
stampaTabella(gender_counts, "NUMERO DI STUDENTI GENERE VS RELATIONSHIP STATUS")


# HEATMAP V DI CRAMER TRA LE VARIABILI CATEGORICHE

cat_cols = categorical_columns + ["Depression"]
cramers_matrix = pd.DataFrame(index=cat_cols, columns=cat_cols, dtype=float)

for a in cat_cols:
    for b in cat_cols:
        tabella = pd.crosstab(df[a], df[b])
        cramers_matrix.loc[a, b] = calcolaCramersV(tabella)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    cramers_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    vmin=0,
    vmax=1,
    ax=ax
)

ax.set_title("V di Cramér tra le variabili categoriche")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
salvaGrafico("plot_vCramer.svg")


# ASSOCIAZIONE VARIABILI NUMERICHE CON DEPRESSIONE

numeric_columns = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
results = []

for colonna in numeric_columns:
    no_group = df.loc[df["Depression"] == "No", colonna].dropna()
    yes_group = df.loc[df["Depression"] == "Yes", colonna].dropna()

    statistic, p_value = mannwhitneyu(
        no_group,
        yes_group,
        alternative="two-sided"
    )

    results.append({
        "Variable": colonna,
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

for colonna in ["P-value", "P-value corrected"]:
    numeric_association_table[colonna] = numeric_association_table[colonna].map(
        formattaPvalue
    )

colonne_arrotondamento = [
    "Median (No)", "Median (Yes)", "U Statistic", "Effect size (A)"
]
numeric_association_table[colonne_arrotondamento] = (
    numeric_association_table[colonne_arrotondamento].round(2)
)

stampaTabella(
    numeric_association_table,
    "ASSOCIAZIONE VARIABILI NUMERICHE CON DEPRESSIONE"
)


# TASSO DI DEPRESSIONE PER QUINTILI DI CGPA

df["CGPA_quintile"] = pd.qcut(
    df["CGPA"],
    q=5,
    labels=["C1 (più basso)", "C2", "C3", "C4", "C5 (più alto)"]
)

cgpa_rates = (
    df.groupby("CGPA_quintile", observed=True)
    .agg(
        CGPA_min=("CGPA", "min"),
        CGPA_max=("CGPA", "max"),
        Students=("Depression", "size"),
        Rate=("Depression", calcolaTasso)
    )
    .reset_index()
)

stampaTabella(cgpa_rates.round(2), "TASSO DI DEPRESSIONE PER QUINTILI DI CGPA")

plotTassoQuintili(
    cgpa_rates,
    "CGPA_quintile",
    "Tasso di depressione per gruppo di CGPA",
    "CGPA (quintili)",
    "plot_cgpa_quintili_depressione.svg"
)


# TEST CHI-QUADRATO E V DI CRAMER PER CGPA

cgpa_table = pd.crosstab(df["CGPA_quintile"], df["Depression"])
chi2_cg, p_cg, dof_cg, expected = chi2_contingency(cgpa_table)
v_cg = calcolaCramersV(cgpa_table)

cgpa_results = []

for quintile in df["CGPA_quintile"].cat.categories:
    mask = df["CGPA_quintile"] == quintile
    tabella = pd.crosstab(mask, df["Depression"])
    p_value = chi2_contingency(tabella)[1]

    cgpa_results.append({
        "Quintile": quintile,
        "Rate (%)": calcolaTasso(df.loc[mask, "Depression"]),
        "Rate others (%)": calcolaTasso(df.loc[~mask, "Depression"]),
        "P-value": p_value,
        "P-value corr.": min(p_value * 5, 1),
        "V di Cramér": np.nan
    })

cgpa_results.extend([
    {"Quintile": "Test complessivo", "P-value": p_cg},
    {"Quintile": "V di Cramér", "V di Cramér": v_cg}
])

cgpa_quintile_table = pd.DataFrame(cgpa_results)

for colonna in ["P-value", "P-value corr."]:
    cgpa_quintile_table[colonna] = cgpa_quintile_table[colonna].map(
        lambda p: formattaPvalue(p) if pd.notna(p) else "-"
    )

stampaTabella(
    cgpa_quintile_table.fillna("-"),
    "TEST CHI-QUADRATO E V DI CRAMER PER CGPA"
)


# CORRELAZIONE DI SPEARMAN

numeric_corr = df[numeric_columns].corr(method="spearman").round(2)
tabella_corr = numeric_corr.copy()
tabella_corr.insert(0, "Metodo", "Spearman")

stampaTabella(tabella_corr, "CORRELAZIONE DI SPEARMAN")

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
salvaGrafico("plot_correlazione_spearman.svg")


# PROFILI DI STUDENTI IN BASE ALLE CATEGORIE

profile_columns = [
    "Relationship_status",
    "friend_circle",
    "Family_background",
    "Living_style"
]

profile_summary = (
    df.groupby(profile_columns, observed=True)
    .agg(
        Total_Students=("Depression", "size"),
        Depression_Count=("Depression", lambda x: (x == "Yes").sum())
    )
    .reset_index()
)

logToFile(f"Profili totali: {len(profile_summary)}")

profile_summary = profile_summary[
    profile_summary["Total_Students"] >= 20
].copy()

logToFile(f"Profili con almeno 20 studenti: {len(profile_summary)}")

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

profile_summary = profile_summary.sort_values(
    "Depression_Rate", ascending=False
)

stampaTabella(
    profile_summary.head(10),
    "PROFILI CON TASSO DI DEPRESSIONE PIU ALTO"
)
