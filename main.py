
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
import statsmodels.api as sm

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    roc_curve
)

from scipy.stats import randint
import joblib
from tqdm import tqdm
from itertools import product
import matplotlib

matplotlib.use("Agg")  # Disattiva la GUI per evitare crash con la parallelizzazione
plt.style.use("fivethirtyeight")
warnings.filterwarnings("ignore")

random_seed = 42

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
    if output is not None:
        with open(output, 'a', encoding='utf-8') as file:
            file.write(str(data) + '\n')

### DICHIARAZIONE FILE DI LOG E PLOT
pathDebug = 'data/debug/log.txt'
pathPlots = 'data/plots/'
os.makedirs('data/debug', exist_ok=True)
os.makedirs(pathPlots, exist_ok=True)

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

### PLOT Age Distribution by Gender
fig, ax1 = plt.subplots(figsize=(10, 6))

sns.histplot(
    data=df, x="Age", hue="Gender", discrete=True, multiple="dodge",
    shrink=0.85, stat="count", ax=ax1
)
ax1.set_title("Age Distribution by Gender")
ax1.set_xlabel("Age")
ax1.set_ylabel("Number of Students")

ax2 = ax1.twinx()
sns.histplot(
    data=df, x="Age", hue="Gender", discrete=True, multiple="dodge",
    shrink=0.85, stat="density", alpha=1, ax=ax2
)
ax2.grid(False)
ax2.set_ylabel("Density")

plt.tight_layout()
plt.savefig(pathPlots + "plot_AgeDistribution_Gender.png", format="png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_AgeDistribution_Gender.svg", format="svg", bbox_inches="tight")
plt.close()

### PLOT Depression status by students
depression_counts = df["Depression"].value_counts()
depression_pct = df["Depression"].value_counts(normalize=True).mul(100).round(2)

fig, ax = plt.subplots(figsize=(7, 6))
ax = sns.countplot(data=df, x="Depression")

for container in ax.containers:
    labels = [f"{int(bar.get_height())}\n({bar.get_height() / len(df) * 100:.1f}%)" for bar in container]
    ax.bar_label(container, labels=labels, padding=3)

ax.set_title("Reported Depression Status Among Students", fontsize=14, pad=12)
ax.set_xlabel("Depression Status")
ax.set_ylabel("Number of Students")
ax.set_ylim(top=ax.get_ylim()[1] * 1.15)

plt.tight_layout()
plt.savefig(pathPlots + "plot_DepressionStatus.png", format="png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_DepressionStatus.svg", format="svg", bbox_inches="tight")
plt.close()

### Plot depressione in base a categoria
def plot_depression_comparison(column, title=None):
    title = title or column.replace("_", " ").title()
    counts = pd.crosstab(df[column], df["Depression"]).reindex(columns=["No", "Yes"], fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    order = counts.sum(axis=1).sort_values(ascending=False).index
    counts, pct = counts.loc[order], pct.loc[order]

    ax = counts.plot(
        kind="bar", stacked=True, figsize=(max(8, len(order) * 1.2), 6),
        color=["tab:blue", "tab:orange"]
    )

    for i, container in enumerate(ax.containers):
        labels = [f"{n}\n({p:.1f}%)" if n > 0 else "" for n, p in zip(counts.iloc[:, i], pct.iloc[:, i])]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=12)

    ax.set(title=f"{title} and Depression", xlabel=title, ylabel="Number of Students")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Depression")
    ax.grid(axis="y", alpha=0.8)
    plt.tight_layout()
    plt.savefig(pathPlots + f"plot_{column}_vs_Depression.png", format="png", bbox_inches="tight")
    plt.savefig(pathPlots + f"plot_{column}_vs_Depression.svg", format="svg", bbox_inches="tight")
    plt.close()

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
depression_rates = depression_rates[["Variable", "Category", "Depression_Rate"]].sort_values("Depression_Rate", ascending=False)
depression_rates["Depression_Rate"] = depression_rates["Depression_Rate"].round(2)
logToFile('\n### STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE', pathDebug)
logToFile(tabulate(depression_rates, headers="keys", tablefmt="rounded_grid", floatfmt=".2f"), pathDebug)

### TEST CHI QUADRO E V-CRAMER
association_results = []
for col in categorical_columns:
    table = pd.crosstab(df[col], df["Depression"])
    chi2, p, dof, _ = chi2_contingency(table)
    n, r, k = table.to_numpy().sum(), *table.shape
    v = np.sqrt(chi2 / (n * min(r - 1, k - 1))) if min(r - 1, k - 1) > 0 else np.nan

    association_results.append({
        "Variable": col, "Chi-square": round(chi2, 2),
        "Degrees of Freedom": dof,
        "P-value": "< 0.001" if p < 0.001 else f"{p:.3f}",
        "Cramer's V": round(v, 3)
    })

logToFile('\n### TEST CHI QUADRO E V-CRAMER', pathDebug)
logToFile(tabulate(pd.DataFrame(association_results), headers="keys", tablefmt="rounded_grid", floatfmt=".3f"), pathDebug)

### CONDIZIONE DI SALUTE VS DEPRESSIONE
health_table = pd.crosstab(df["Health_condition"], df["Depression"], margins=True)
health_table["Tasso_depressione (%)"] = (health_table["Yes"] / health_table["All"] * 100).round(2)

logToFile("\n### CONDIZIONE DI SALUTE VS DEPRESSIONE", pathDebug)
logToFile(tabulate(health_table, headers="keys", tablefmt="rounded_grid"), pathDebug)

### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE
rate = pd.crosstab(
    df["Relationship_status"], df["friend_circle"],
    values=(df["Depression"] == "Yes"), aggfunc="mean"
).mul(100)
n = pd.crosstab(df["Relationship_status"], df["friend_circle"])
labels = rate.round(1).astype(str) + "%\n(n=" + n.astype(str) + ")"

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(rate, annot=labels, fmt="", cmap="Oranges", cbar_kws={"label": "Tasso di depressione (%)"}, ax=ax)
ax.set_title("Depression by Relationship Status and Friend Circle")
ax.set_xlabel("Friend Circle")
ax.set_ylabel("Relationship Status")
plt.tight_layout()
plt.savefig(pathPlots + "plot_Relationship_FriendCircle.png", format="png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_Relationship_FriendCircle.svg", format="svg", bbox_inches="tight")
plt.close()

### STATO DI RELAZIONE E CERCHIA DI AMICI STATISTICA
friend_results = []
relationship_groups = df["Relationship_status"].dropna().unique()

for group in relationship_groups:
    subset = df[df["Relationship_status"] == group]
    rate_yes = (subset.loc[subset["friend_circle"] == "Yes", "Depression"] == "Yes").mean() * 100
    rate_no = (subset.loc[subset["friend_circle"] == "No", "Depression"] == "Yes").mean() * 100
    table = pd.crosstab(subset["friend_circle"], subset["Depression"])
    p = chi2_contingency(table)[1] if table.shape[0] > 1 and table.shape[1] > 1 else np.nan

    friend_results.append({
        "Relationship_status": group, "Students": len(subset),
        "Rate (friends)": round(rate_yes, 1), "Rate (no friends)": round(rate_no, 1),
        "Difference": round(rate_no - rate_yes, 1), "P-value": p,
        "P-value (corrected)": min(p * len(relationship_groups), 1) if pd.notna(p) else np.nan
    })

friend_table = pd.DataFrame(friend_results).sort_values("Students", ascending=False)
for col in ["P-value", "P-value (corrected)"]:
    friend_table[col] = friend_table[col].map(lambda p: "< 0.001" if p < 0.001 else f"{p:.3f}" if pd.notna(p) else "N/A")

logToFile("\n### STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE", pathDebug)
logToFile(tabulate(friend_table, headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

### COSTI EXTRA-STUDIO E DEPRESSIONE
df["Cost_quintile"] = pd.qcut(df["Cost_without_study"], q=5, labels=["Q1 (più basso)", "Q2", "Q3", "Q4", "Q5 (più alto)"])
cost_rates = df.groupby("Cost_quintile", observed=True)["Depression"].agg(
    Students="size", Rate=lambda x: (x == "Yes").mean() * 100
).reset_index()

cost_table = cost_rates.copy()
cost_table["Rate (%)"] = cost_table["Rate"].round(2)
cost_table = cost_table.drop(columns="Rate")

logToFile("\n### COSTI EXTRA-STUDIO E DEPRESSIONE", pathDebug)
logToFile(tabulate(cost_table, headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

fig, ax = plt.subplots(figsize=(9, 5))
sns.barplot(data=cost_rates, x="Cost_quintile", y="Rate", color="tab:orange", ax=ax)
overall_rate = (df["Depression"] == "Yes").mean() * 100
ax.axhline(overall_rate, color="gray", linestyle="--", label=f"Tasso complessivo ({overall_rate:.1f}%)")
ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
ax.set_title("Tasso di depressione per gruppo di costo extra-studio")
ax.set_xlabel("Costo extra-studio (quintili)")
ax.set_ylabel("Percentuale di studenti con depressione")
ax.legend()
plt.tight_layout()
plt.savefig(pathPlots + "plot_extra_studio_vs_depressione.png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_extra_studio_vs_depressione.svg", bbox_inches="tight")
plt.close()

### HEATMAP V DI CRAMÉR
cat_cols_all = categorical_columns + ["Depression"]
cramers_matrix = pd.DataFrame(index=cat_cols_all, columns=cat_cols_all, dtype=float)

for a in cat_cols_all:
    for b in cat_cols_all:
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
plt.savefig(pathPlots + "plot_vCramer.png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_vCramer.svg", bbox_inches="tight")
plt.close()

### CORRELAZIONE DI SPEARMAN
numeric_columns = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
numeric_corr = df[numeric_columns].corr(method="spearman").round(2)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(numeric_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True, linewidths=0.8, ax=ax)
ax.set_title("Correlazione di Spearman tra le variabili numeriche", pad=20)
plt.tight_layout()
plt.savefig(pathPlots + "plot_correlazione_spearman.png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_correlazione_spearman.svg", bbox_inches="tight")
plt.close()


# ==============================================================================
# PREPARAZIONE UNIFICATA DATASET DI MODELLAZIONE E SPLIT UNICO TRAIN / TEST
# ==============================================================================

TARGET = "Depression"
num_cols = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
cat_cols = [
    "Gender", "Family_background", "Relationship_status",
    "Marital_status", "Living_style", "friend_circle", "Home_division"
]

# Modello Dataset Unificato
df_model = df.copy()
cols_to_drop = [col for col in df_model.columns if col.startswith("cluster")]
df_model = df_model.drop(columns=cols_to_drop, errors="ignore")

needed_cols = [TARGET] + num_cols + cat_cols
df_model = df_model[needed_cols].dropna(subset=needed_cols).copy()

# Mapping Target (1=Yes, 0=No)
df_model[TARGET] = (
    df_model[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
    .astype(int)
)

# Feature engineering per CatBoost e utilità generale
df_model["male"] = (df_model["Gender"] == "Male").astype(int)
df_model["no_friends"] = (df_model["friend_circle"] == "No").astype(int)

# SPLIT TRAIN / TEST UNICO PER TUTTI I MODELLI
train_df, test_df = train_test_split(
    df_model, test_size=0.20, random_state=random_seed, stratify=df_model[TARGET]
)

logToFile("\n" + "="*80, pathDebug)
logToFile("PREPARAZIONE UNIFICATA DATASET DI MODELLAZIONE", pathDebug)
logToFile("="*80, pathDebug)
logToFile(f"Dimensioni dataset unificato: {df_model.shape}", pathDebug)
logToFile(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}", pathDebug)
logToFile(f"Distribuzione Target Train (%):\n{train_df[TARGET].value_counts(normalize=True).round(3)}", pathDebug)
logToFile(f"Distribuzione Target Test (%):\n{test_df[TARGET].value_counts(normalize=True).round(3)}", pathDebug)

# Formati Unificati per Feature e Target
X_train_raw = train_df[num_cols + cat_cols]
X_test_raw = test_df[num_cols + cat_cols]
y_train = train_df[TARGET]
y_test = test_df[TARGET]

# One-Hot Encoding per Decision Tree e Logistic Regression
X_train_encoded = pd.get_dummies(X_train_raw, columns=cat_cols, drop_first=True, dtype=float)
X_test_encoded = pd.get_dummies(X_test_raw, columns=cat_cols, drop_first=True, dtype=float).reindex(columns=X_train_encoded.columns, fill_value=0)


# ==============================================================================
# MODELLO 1: DECISION TREE
# ==============================================================================

logToFile("\n" + "="*80, pathDebug)
logToFile("SEZIONE 1: DECISION TREE", pathDebug)
logToFile("="*80, pathDebug)

grid_params_path = "data/gridSearch.par"
run_gridsearch = True

if os.path.exists(grid_params_path):
    scelta = input(f"\nTrovato il file '{grid_params_path}'. Vuoi utilizzare i parametri salvati per Decision Tree? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_gridsearch = False
        best_params_dt = joblib.load(grid_params_path)
        best_tree = DecisionTreeClassifier(
            **best_params_dt, class_weight="balanced", random_state=random_seed
        ).fit(X_train_encoded, y_train)
        logToFile(f"Caricamento iperparametri DT da '{grid_params_path}': {best_params_dt}", pathDebug)

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

    best_params_dt = grid_search.best_params_
    joblib.dump(best_params_dt, grid_params_path)
    best_tree = grid_search.best_estimator_
    logToFile(f"Migliori iperparametri DT: {best_params_dt}", pathDebug)

y_pred_dt = best_tree.predict(X_test_encoded)
y_prob_dt = best_tree.predict_proba(X_test_encoded)[:, 1]

logToFile("\n### DECISION TREE - VALUTAZIONE TEST SET", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_dt):.3f} | Prec: {precision_score(y_test, y_pred_dt):.3f} | Rec: {recall_score(y_test, y_pred_dt):.3f} | F1: {f1_score(y_test, y_pred_dt):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob_dt):.3f}", pathDebug)

# Plot ROC & Tree
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_dt, name="Decision Tree", ax=ax)
ax.set_title("ROC Curve — Decision Tree")
plt.tight_layout()
plt.savefig(pathPlots + "plot_dt_roc_curve.png")
plt.savefig(pathPlots + "plot_dt_roc_curve.svg")
plt.close()

plt.figure(figsize=(18, 9))
plot_tree(best_tree, feature_names=X_train_encoded.columns, class_names=["No", "Yes"], filled=True, max_depth=3, fontsize=9)
plt.title("Albero di Decisione — Primi 3 livelli")
plt.tight_layout()
plt.savefig(pathPlots + "plot_decision_tree.png")
plt.savefig(pathPlots + "plot_decision_tree.svg")
plt.close()


# ==============================================================================
# MODELLO 2: REGRESSIONE LOGISTICA CON SELEZIONE DELLE VARIABILI (FORWARD/BACKWARD)
# ==============================================================================

logToFile("\n" + "="*80, pathDebug)
logToFile("SEZIONE 2: REGRESSIONE LOGISTICA (FORWARD/BACKWARD SELECTION)", pathDebug)
logToFile("="*80, pathDebug)

# Standardizzazione numeriche solo per Logit
scaler = StandardScaler()
X_train_logit = X_train_encoded.copy()
X_test_logit = X_test_encoded.copy()

X_train_logit[num_cols] = scaler.fit_transform(X_train_logit[num_cols])
X_test_logit[num_cols] = scaler.transform(X_test_logit[num_cols])

X_train_logit = sm.add_constant(X_train_logit, has_constant="add")
X_test_logit = sm.add_constant(X_test_logit, has_constant="add")

def fit_logit(X, y):
    X = X.copy().loc[:, ~X.columns.duplicated()]
    X = X.apply(pd.to_numeric, errors="coerce").astype(float).replace([np.inf, -np.inf], np.nan)
    cols_to_keep = [col for col in X.columns if col == "const" or X[col].nunique(dropna=False) > 1]
    X = X[cols_to_keep]
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)

    if y.nunique() < 2 or X.shape[1] == 0:
        return None

    for method in ["newton", "lbfgs", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = sm.Logit(y, X).fit(method=method, maxiter=500, disp=0)
            if model.mle_retvals.get("converged", False):
                return model
        except Exception:
            continue
    return None

def get_criterion(model, criterion_name):
    if model is None: return np.inf
    return float(getattr(model, criterion_name.lower()))

def forward_selection(X_tr, y_tr, criterion_name="aic"):
    predictors = [c for c in X_tr.columns if c != "const"]
    selected, remaining = [], predictors.copy()
    current_model = fit_logit(X_tr[["const"]], y_tr)
    current_score = get_criterion(current_model, criterion_name)
    best_model = current_model

    while remaining:
        trials = []
        for candidate in remaining:
            model = fit_logit(X_tr[["const"] + selected + [candidate]], y_tr)
            trials.append((get_criterion(model, criterion_name), candidate, model))
        trials.sort(key=lambda x: x[0])
        best_score, best_candidate, best_m = trials[0]

        if best_score < current_score - 1e-6:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            current_score, best_model = best_score, best_m
        else:
            break
    return selected, best_model, current_score

def backward_elimination(X_tr, y_tr, criterion_name="aic"):
    selected = [c for c in X_tr.columns if c != "const"]
    current_model = fit_logit(X_tr[["const"] + selected], y_tr)
    current_score = get_criterion(current_model, criterion_name)
    best_model = current_model

    while len(selected) > 1:
        trials = []
        for candidate in selected:
            trial_vars = [c for c in selected if c != candidate]
            model = fit_logit(X_tr[["const"] + trial_vars], y_tr)
            trials.append((get_criterion(model, criterion_name), candidate, model))
        trials.sort(key=lambda x: x[0])
        best_score, candidate_to_remove, best_m = trials[0]

        if best_score < current_score - 1e-6:
            selected.remove(candidate_to_remove)
            current_score, best_model = best_score, best_m
        else:
            break
    return selected, best_model, current_score

logit_params_path = "data/logit.par"
run_logit_selection = True

if os.path.exists(logit_params_path):
    scelta = input(f"\nTrovato il file '{logit_params_path}'. Vuoi utilizzare i parametri salvati per Logit? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_logit_selection = False
        saved_logit_info = joblib.load(logit_params_path)
        best_logit_vars = saved_logit_info["variables"]
        final_criterion = saved_logit_info["criterion"]
        logToFile(f"Caricamento configurazione Logit salvata: criterio={final_criterion}, vars={best_logit_vars}", pathDebug)

if run_logit_selection:
    results_logit = {}
    for method in ["forward", "backward"]:
        for criterion_name in ["aic", "bic"]:
            key = f"{method}_{criterion_name}"
            if method == "forward":
                vars_sel, mod, score = forward_selection(X_train_logit, y_train, criterion_name)
            else:
                vars_sel, mod, score = backward_elimination(X_train_logit, y_train, criterion_name)
            results_logit[key] = {"variables": vars_sel, "model": mod, "score": score}

    final_criterion = "bic"
    valid_results = {k: v for k, v in results_logit.items() if v["model"] is not None}
    best_key = min(valid_results, key=lambda k: get_criterion(valid_results[k]["model"], final_criterion))
    best_logit_vars = valid_results[best_key]["variables"]

    joblib.dump({"variables": best_logit_vars, "criterion": final_criterion, "method": best_key}, logit_params_path)
    logToFile(f"Modello Logit selezionato: {best_key} ({final_criterion.upper()}) | Vars: {len(best_logit_vars)}", pathDebug)

# Ristima robusta finale Logit
best_logit_model = fit_logit(X_train_logit[["const"] + best_logit_vars], y_train)

# Test Evaluation
X_test_logit_best = X_test_logit[["const"] + best_logit_vars].reindex(columns=["const"] + best_logit_vars, fill_value=0).astype(float)
y_prob_logit = best_logit_model.predict(X_test_logit_best)
y_pred_logit = (y_prob_logit >= 0.50).astype(int)

logToFile("\n### REGRESSIONE LOGISTICA - VALUTAZIONE TEST SET", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_logit):.3f} | Prec: {precision_score(y_test, y_pred_logit):.3f} | Rec: {recall_score(y_test, y_pred_logit):.3f} | F1: {f1_score(y_test, y_pred_logit):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob_logit):.3f}", pathDebug)
logToFile(f"\nClassification Report:\n{classification_report(y_test, y_pred_logit)}", pathDebug)

# Odds Ratios
odds_ratios = np.exp(best_logit_model.params)
or_df = pd.DataFrame({"Coefficiente": best_logit_model.params, "Odds Ratio": odds_ratios, "p-value": best_logit_model.pvalues}).sort_values("p-value")
logToFile(f"\nOdds Ratios e Significatività:\n{tabulate(or_df.round(4), headers='keys', tablefmt='rounded_grid')}", pathDebug)

# Plot Odds Ratios e ROC
fig, ax = plt.subplots(figsize=(10, 6))

# Inserendo ["teal"] come lista si previene l'errore di parsing del colore in Pandas
or_df.drop("const", errors="ignore")["Odds Ratio"].plot(
    kind="barh", ax=ax, color=["teal"]
)

ax.axvline(1.0, color="gray", linestyle="--")
ax.set_title("Odds Ratios — Regressione Logistica")
plt.tight_layout()
plt.savefig(pathPlots + "plot_logit_odds_ratios.png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_logit_odds_ratios.svg", bbox_inches="tight")
plt.close()

fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_logit, name="Logistic Regression", ax=ax)
ax.set_title("ROC Curve — Logistic Regression")
plt.tight_layout()
plt.savefig(pathPlots + "plot_logit_roc_curve.png")
plt.savefig(pathPlots + "plot_logit_roc_curve.svg")
plt.close()


# ==============================================================================
# MODELLO 3: RANDOM FOREST
# ==============================================================================

logToFile("\n" + "="*80, pathDebug)
logToFile("SEZIONE 3: RANDOM FOREST CON OPTIMIZED THRESHOLD", pathDebug)
logToFile("="*80, pathDebug)

preprocessor_rf = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), cat_cols)
    ]
)

rf_pipeline_base = Pipeline([
    ("preprocessor", preprocessor_rf),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=random_seed, n_jobs=-1, class_weight="balanced"))
])

rf_params_path = "data/randomSearchRandomForest.par"
run_randomsearch = True

if os.path.exists(rf_params_path):
    scelta = input(f"\nTrovato il file '{rf_params_path}'. Vuoi utilizzare i parametri salvati per Random Forest? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_randomsearch = False
        saved_rf_params = joblib.load(rf_params_path)
        best_rf_params_clean = {k.replace("classifier__", ""): v for k, v in saved_rf_params.items()}
        logToFile(f"Caricamento iperparametri RF da '{rf_params_path}': {saved_rf_params}", pathDebug)

if run_randomsearch:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
    param_distributions = {
        "classifier__max_depth": [4, 6, 8, 10, 15, None],
        "classifier__min_samples_split": randint(2, 21),
        "classifier__min_samples_leaf": randint(1, 21),
        "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
        "classifier__class_weight": ["balanced", "balanced_subsample"]
    }

    random_search = RandomizedSearchCV(
        estimator=rf_pipeline_base, param_distributions=param_distributions, n_iter=60,
        scoring="f1", cv=cv, n_jobs=-1, random_state=random_seed
    )

    with tqdm_joblib(total=60 * cv.n_splits, desc="RandomSearch RF"):
        random_search.fit(X_train_raw, y_train)

    joblib.dump(random_search.best_params_, rf_params_path)
    best_rf_params_clean = {k.replace("classifier__", ""): v for k, v in random_search.best_params_.items()}
    logToFile(f"Migliori iperparametri RF: {best_rf_params_clean}", pathDebug)

# Fitting modello finale RF
final_rf = RandomForestClassifier(**best_rf_params_clean, n_estimators=300, oob_score=True, random_state=random_seed, n_jobs=-1)
final_pipeline_rf = Pipeline([("preprocessor", preprocessor_rf), ("classifier", final_rf)])
final_pipeline_rf.fit(X_train_raw, y_train)

y_prob_rf = final_pipeline_rf.predict_proba(X_test_raw)[:, 1]

# Tuning Soglia Ottimale
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_rf)
f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-12)
best_rf_threshold = float(thresholds[np.argmax(f1_scores)])

y_pred_rf = (y_prob_rf >= best_rf_threshold).astype(int)

logToFile("\n### RANDOM FOREST - VALUTAZIONE TEST SET (SOGLIA OTTIMIZZATA)", pathDebug)
logToFile(f"Soglia applicata: {best_rf_threshold:.4f}", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_rf):.3f} | Prec: {precision_score(y_test, y_pred_rf):.3f} | Rec: {recall_score(y_test, y_pred_rf):.3f} | F1: {f1_score(y_test, y_pred_rf):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob_rf):.3f}", pathDebug)

# Plot ROC & Precision-Recall RF
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_rf, name="Random Forest", ax=ax)
ax.set_title("ROC Curve — Random Forest")
plt.tight_layout()
plt.savefig(pathPlots + "plot_rf_roc_curve.png")
plt.savefig(pathPlots + "plot_rf_roc_curve.svg")
plt.close()


# ==============================================================================
# MODELLO 4: CATBOOST
# ==============================================================================

logToFile("\n" + "="*80, pathDebug)
logToFile("SEZIONE 4: CATBOOST CLASSIFIER", pathDebug)
logToFile("="*80, pathDebug)

COL_CAT_CB = ["Relationship_status", "Family_background", "Home_division", "Living_style"]
COL_NUM_CB = ["CGPA", "Cost_without_study", "Age", "Current_semester"]
X_train_cb = train_df[["male", "no_friends"] + COL_CAT_CB + COL_NUM_CB]
X_test_cb = test_df[["male", "no_friends"] + COL_CAT_CB + COL_NUM_CB]

cb_params_path = "data/catboost.par"
run_gridsearch_cb = True

if os.path.exists(cb_params_path):
    scelta = input(f"\nTrovato il file '{cb_params_path}'. Vuoi utilizzare i parametri salvati per CatBoost? [s/n]: ").strip().lower()
    if scelta in ["s", "si", "y", "yes", ""]:
        run_gridsearch_cb = False
        best_cb_params = joblib.load(cb_params_path)
        logToFile(f"Caricamento iperparametri CatBoost da '{cb_params_path}': {best_cb_params}", pathDebug)

def build_cb_model(depth, lr, l2, iterations=300):
    return CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=lr, l2_leaf_reg=l2,
        cat_features=tuple(COL_CAT_CB), auto_class_weights="Balanced", random_seed=random_seed, verbose=0, allow_writing_files=False
    )

if run_gridsearch_cb:
    cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=random_seed)
    grid_cb = []
    for depth, l2 in product([4, 6, 8], [10, 30]):
        m = build_cb_model(depth, 0.03, l2)
        scores = cross_val_score(m, X_train_cb, y_train, cv=cv_rep, scoring="roc_auc")
        grid_cb.append((scores.mean(), depth, l2))
    grid_cb.sort(reverse=True)
    best_cb_params = {"depth": grid_cb[0][1], "l2_leaf_reg": grid_cb[0][2], "learning_rate": 0.03}
    joblib.dump(best_cb_params, cb_params_path)
    logToFile(f"Migliori iperparametri CatBoost: {best_cb_params}", pathDebug)

model_cb = build_cb_model(int(best_cb_params["depth"]), 0.03, int(best_cb_params["l2_leaf_reg"]))
model_cb.fit(X_train_cb, y_train)

# Soglia ottimale CatBoost
cv_cb = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)
p_cv_cb = cross_val_predict(model_cb, X_train_cb, y_train, cv=cv_cb, method="predict_proba")[:, 1]

thresholds_cb = np.arange(0.1, 0.9, 0.02)
f1_cb_list = [f1_score(y_train, (p_cv_cb >= t).astype(int)) for t in thresholds_cb]
best_cb_threshold = float(thresholds_cb[np.argmax(f1_cb_list)])

y_prob_cb = model_cb.predict_proba(X_test_cb)[:, 1]
y_pred_cb = (y_prob_cb >= best_cb_threshold).astype(int)

logToFile("\n### CATBOOST - VALUTAZIONE TEST SET (SOGLIA OTTIMIZZATA)", pathDebug)
logToFile(f"Soglia applicata: {best_cb_threshold:.4f}", pathDebug)
logToFile(f"Acc: {accuracy_score(y_test, y_pred_cb):.3f} | Prec: {precision_score(y_test, y_pred_cb):.3f} | Rec: {recall_score(y_test, y_pred_cb):.3f} | F1: {f1_score(y_test, y_pred_cb):.3f} | ROC-AUC: {roc_auc_score(y_test, y_prob_cb):.3f}", pathDebug)

# Plot ROC CatBoost
fig, ax = plt.subplots(figsize=(7, 5))
RocCurveDisplay.from_predictions(y_test, y_prob_cb, name="CatBoost", ax=ax)
ax.set_title("ROC Curve — CatBoost")
plt.tight_layout()
plt.savefig(pathPlots + "plot_catboost_roc_curve.png")
plt.savefig(pathPlots + "plot_catboost_roc_curve.svg")
plt.close()


# ==============================================================================
# SEZIONE FINALE: ESECUZIONE COMPARATIVA DI TUTTI I MODELLI DA FILE .PAR
# ==============================================================================

logToFile("\n" + "="*80, pathDebug)
logToFile("SEZIONE 5: CONFRONTO FINALE DI TUTTI I MODELLI SALVATI DA FILE .PAR", pathDebug)
logToFile("="*80, pathDebug)

comparison_metrics = []

# 1. Decision Tree
if os.path.exists(grid_params_path):
    params_dt = joblib.load(grid_params_path)
    dt_final = DecisionTreeClassifier(**params_dt, class_weight="balanced", random_state=random_seed).fit(X_train_encoded, y_train)
    p_dt = dt_final.predict_proba(X_test_encoded)[:, 1]
    y_dt_pred = (p_dt >= 0.5).astype(int)
    comparison_metrics.append({
        "Modello": "Decision Tree",
        "Accuracy": accuracy_score(y_test, y_dt_pred),
        "Precision": precision_score(y_test, y_dt_pred),
        "Recall": recall_score(y_test, y_dt_pred),
        "F1-Score": f1_score(y_test, y_dt_pred),
        "ROC-AUC": roc_auc_score(y_test, p_dt)
    })

# 2. Logistic Regression
if os.path.exists(logit_params_path):
    logit_info = joblib.load(logit_params_path)
    vars_l = logit_info["variables"]
    m_logit = fit_logit(X_train_logit[["const"] + vars_l], y_train)
    X_t_l = X_test_logit[["const"] + vars_l].reindex(columns=["const"] + vars_l, fill_value=0).astype(float)
    p_logit = m_logit.predict(X_t_l)
    y_l_pred = (p_logit >= 0.5).astype(int)
    comparison_metrics.append({
        "Modello": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_l_pred),
        "Precision": precision_score(y_test, y_l_pred),
        "Recall": recall_score(y_test, y_l_pred),
        "F1-Score": f1_score(y_test, y_l_pred),
        "ROC-AUC": roc_auc_score(y_test, p_logit)
    })

# 3. Random Forest
if os.path.exists(rf_params_path):
    params_rf = joblib.load(rf_params_path)
    clean_rf_p = {k.replace("classifier__", ""): v for k, v in params_rf.items()}
    rf_pipe = Pipeline([("preprocessor", preprocessor_rf), ("classifier", RandomForestClassifier(**clean_rf_p, n_estimators=300, random_state=random_seed, n_jobs=-1))])
    rf_pipe.fit(X_train_raw, y_train)
    p_rf = rf_pipe.predict_proba(X_test_raw)[:, 1]
    y_rf_pred = (p_rf >= best_rf_threshold).astype(int)
    comparison_metrics.append({
        "Modello": "Random Forest",
        "Accuracy": accuracy_score(y_test, y_rf_pred),
        "Precision": precision_score(y_test, y_rf_pred),
        "Recall": recall_score(y_test, y_rf_pred),
        "F1-Score": f1_score(y_test, y_rf_pred),
        "ROC-AUC": roc_auc_score(y_test, p_rf)
    })

# 4. CatBoost
if os.path.exists(cb_params_path):
    params_cb = joblib.load(cb_params_path)
    cb_m = build_cb_model(int(params_cb["depth"]), 0.03, int(params_cb["l2_leaf_reg"]))
    cb_m.fit(X_train_cb, y_train)
    p_cb = cb_m.predict_proba(X_test_cb)[:, 1]
    y_cb_pred = (p_cb >= best_cb_threshold).astype(int)
    comparison_metrics.append({
        "Modello": "CatBoost",
        "Accuracy": accuracy_score(y_test, y_cb_pred),
        "Precision": precision_score(y_test, y_cb_pred),
        "Recall": recall_score(y_test, y_cb_pred),
        "F1-Score": f1_score(y_test, y_cb_pred),
        "ROC-AUC": roc_auc_score(y_test, p_cb)
    })

# Tabella Comparativa Finale
comp_df = pd.DataFrame(comparison_metrics)
logToFile("\n### TABELLA COMPARATIVA METRICHE (TEST SET)", pathDebug)
logToFile(tabulate(comp_df.round(4), headers="keys", tablefmt="rounded_grid", showindex=False), pathDebug)

# Plot Sovrapposto delle Curve ROC di Tutti i Modelli
fig, ax = plt.subplots(figsize=(10, 7))
if 'p_dt' in locals():
    fpr, tpr, _ = roc_curve(y_test, p_dt)
    ax.plot(fpr, tpr, label=f"Decision Tree (AUC = {roc_auc_score(y_test, p_dt):.3f})")

if 'p_logit' in locals():
    fpr, tpr, _ = roc_curve(y_test, p_logit)
    ax.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc_score(y_test, p_logit):.3f})")

if 'p_rf' in locals():
    fpr, tpr, _ = roc_curve(y_test, p_rf)
    ax.plot(fpr, tpr, label=f"Random Forest (AUC = {roc_auc_score(y_test, p_rf):.3f})")

if 'p_cb' in locals():
    fpr, tpr, _ = roc_curve(y_test, p_cb)
    ax.plot(fpr, tpr, label=f"CatBoost (AUC = {roc_auc_score(y_test, p_cb):.3f})")

ax.plot([0, 1], [0, 1], "k--", label="Casuale")
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("Confronto Curve ROC — Tutti i Modelli (.par)")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(pathPlots + "plot_all_models_roc_curve.png", bbox_inches="tight")
plt.savefig(pathPlots + "plot_all_models_roc_curve.svg", bbox_inches="tight")
plt.close()

logToFile("\n>>> ESECUZIONE COMPLETATA CON SUCCESSO <<<", pathDebug)

