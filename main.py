### ============================================================
### IMPORT E CONFIGURAZIONE
### ============================================================
import os
import warnings
from itertools import product

os.environ["PYTHONWARNINGS"] = "ignore"  # vale anche per i worker joblib/loky
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")  # niente GUI: evita crash con la parallelizzazione
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from catboost import CatBoostClassifier
from scipy.stats import chi2_contingency, mannwhitneyu, randint
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold, StratifiedKFold,
    cross_val_predict, cross_val_score, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from statsmodels.stats.proportion import proportion_confint
from tabulate import tabulate
from tqdm import tqdm

SEED = 42
TARGET = "Depression"
DATA_FILE = "Student_Depression_Dataset.csv"
LOG, PLOTS = "data/debug/log.txt", "data/plots/"
PAR_DIR = "data/"  # file .par con gli iperparametri salvati (gridSearch, randomSearchRandomForest, catboost, logit)
NUM = ["Age", "Current_semester", "CGPA", "Cost_without_study"]

plt.style.use("fivethirtyeight")
os.makedirs(os.path.dirname(LOG), exist_ok=True)
os.makedirs(PLOTS, exist_ok=True)
if os.path.exists(LOG):
    os.remove(LOG)

model_results = {}  # riempito da evaluate(), usato nel confronto finale


### ============================================================
### FUNZIONI DI SUPPORTO
### ============================================================
class tqdm_joblib:
    """Progress bar per GridSearchCV / RandomizedSearchCV con n_jobs=-1."""
    def __init__(self, total, desc):
        self.pbar = tqdm(total=total, desc=desc)
        self.old_cb = joblib.parallel.BatchCompletionCallBack
        pbar = self.pbar

        class CB(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                pbar.update(n=self.batch_size)
                super().__call__(*args, **kwargs)
        joblib.parallel.BatchCompletionCallBack = CB

    def __enter__(self):
        return self

    def __exit__(self, *args):
        joblib.parallel.BatchCompletionCallBack = self.old_cb
        self.pbar.close()


def log(data):
    print(data)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(str(data) + "\n")


def log_table(title, data, floatfmt=".2f", index=False, **kw):
    log(f"\n### {title}")
    log(tabulate(data, headers="keys", tablefmt="rounded_grid", floatfmt=floatfmt, showindex=index, **kw))


def save(name, svg=False, **kw):
    plt.tight_layout()
    plt.savefig(f"{PLOTS}{name}.png", bbox_inches="tight", **kw)
    if svg:
        plt.savefig(f"{PLOTS}{name}.svg", bbox_inches="tight")
    plt.close()


def fmt_p(p):
    return "-" if pd.isna(p) else "< 0.001" if p < 0.001 else f"{p:.3f}"


def fmt_p_cols(t, cols):
    t = t.copy()
    for c in cols:
        t[c] = t[c].map(fmt_p)
    return t


def assoc(table):
    """Chi-quadro + V di Cramér su una tabella di contingenza -> (chi2, p, dof, V)."""
    chi2, p, dof, _ = chi2_contingency(table)
    k = min(table.shape) - 1
    return chi2, p, dof, (np.sqrt(chi2 / (table.to_numpy().sum() * k)) if k > 0 else np.nan)


def rate_ct(a, b, data=None):
    """Tasso di depressione (%) per ogni combinazione delle colonne a (righe) e b (colonne)."""
    data = df if data is None else data
    return pd.crosstab(data[a], data[b], values=data["dep"], aggfunc="mean").mul(100)


def load_or_search(path, search):
    """Carica gli iperparametri salvati (se l'utente accetta) oppure lancia search() e li salva."""
    if os.path.exists(path) and input(
        f"\nTrovato il file '{path}'. Vuoi utilizzare i parametri salvati? [s/n]: "
    ).strip().lower() in ("s", "si", "y", "yes", ""):
        params = joblib.load(path)
    else:
        params = search()
        joblib.dump(params, path)
    log(f"\n### IPERPARAMETRI ({path})\n{params}")
    return params


def split(cols, label):
    """Train/test stratificato 80/20 su df[cols] (righe incomplete escluse)."""
    d = df[cols + ["dep"]].dropna()
    X_tr, X_te, y_tr, y_te = train_test_split(
        d[cols], d["dep"], test_size=0.2, random_state=SEED, stratify=d["dep"]
    )
    dist = pd.concat([y_tr.value_counts(normalize=True), y_te.value_counts(normalize=True)],
                     axis=1, keys=["train", "test"]).round(3)
    log(f"\n### DATASET {label}\nDimensioni: {d.shape} | Train: {X_tr.shape} | Test: {X_te.shape}")
    log(f"Distribuzione target:\n{dist}")
    return X_tr, X_te, y_tr, y_te


def evaluate(name, y_true, prob, thr=0.5, params=None, store=True):
    """Logga le metriche sul test set e (se store) salva i risultati per il confronto finale."""
    y_true, prob = np.asarray(y_true), np.asarray(prob)
    pred = (prob >= thr).astype(int)
    log(f"\n### {name.upper()} - TEST SET (soglia {thr:.3f})")
    log(f"Acc: {accuracy_score(y_true, pred):.3f} | Prec: {precision_score(y_true, pred):.3f} | "
        f"Rec: {recall_score(y_true, pred):.3f} | F1: {f1_score(y_true, pred):.3f} | "
        f"ROC-AUC: {roc_auc_score(y_true, prob):.3f}")
    cm = tabulate(confusion_matrix(y_true, pred), headers=["Pred No (0)", "Pred Yes (1)"], tablefmt="rounded_grid")
    log(f"\nConfusion matrix:\n{cm}\n\nClassification report:\n{classification_report(y_true, pred)}")
    if store:
        model_results[name] = dict(y_true=y_true, y_prob=prob, y_pred=pred,
                                   threshold=float(thr), params=dict(params or {}))
    return pred


def plot_importance(name, names, values, cmap, fname):
    imp = (pd.DataFrame({"Feature": names, "Importance": values})
           .sort_values("Importance", ascending=False).head(15))
    log_table(f"FEATURE IMPORTANCE {name.upper()} (TOP 15)", imp, floatfmt=".4f")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=imp, x="Importance", y="Feature", hue="Feature", palette=cmap, legend=False)
    plt.title(f"Feature importance — {name}")
    save(fname, svg=True)


### ============================================================
### CARICAMENTO E QUALITÀ DEI DATI
### ============================================================
df = pd.read_csv(DATA_FILE)
assert df[TARGET].isin(["Yes", "No"]).all(), "Il target contiene valori diversi da Yes/No."

log_table("ANALISI DATI MANCANTI", df.isna().sum().to_frame("Valori mancanti"), index=True)
log_table("ANALISI STATISTICA COLONNE NUMERICHE", df.describe(), index=True)
log_table("RELATIONSHIP STATUS VS MARITAL STATUS",
          pd.crosstab(df["Relationship_status"], df["Marital_status"]), index=True)

categorical_columns = [c for c in df.select_dtypes(include=["object", "category", "string"]).columns
                       if c != TARGET]
df["dep"] = (df[TARGET] == "Yes").astype(int)  # target numerico per tassi e modelli
OVERALL = df["dep"].mean() * 100


### ============================================================
### ANALISI ESPLORATIVA: GRAFICI
### ============================================================
# Distribuzione età per genere (conteggi a sinistra, densità a destra)
fig, ax1 = plt.subplots(figsize=(10, 6))
hist_kw = dict(data=df, x="Age", hue="Gender", discrete=True, multiple="dodge", shrink=0.85)
sns.histplot(**hist_kw, stat="count", ax=ax1)
ax2 = ax1.twinx()
sns.histplot(**hist_kw, stat="density", alpha=1, ax=ax2)
ax2.grid(False)
ax1.set(title="Age Distribution by Gender", xlabel="Age", ylabel="Number of Students")
ax2.set_ylabel("Density")
save("plot_AgeDistribution_Gender")

# Stato di depressione complessivo
fig, ax = plt.subplots(figsize=(7, 6))
sns.countplot(data=df, x=TARGET, ax=ax)
ax.bar_label(ax.containers[0],
             labels=[f"{int(b.get_height())}\n({b.get_height() / len(df) * 100:.1f}%)" for b in ax.containers[0]],
             padding=3)
ax.set_title("Reported Depression Status Among Students", fontsize=14, pad=12)
ax.set(xlabel="Depression Status", ylabel="Number of Students", ylim=(0, ax.get_ylim()[1] * 1.15))
save("plot_DepressionStatus")


def plot_depression_comparison(name):
    title = name.replace("_", " ").title()
    counts = pd.crosstab(df[name], df[TARGET]).reindex(columns=["No", "Yes"], fill_value=0)
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]
    pct = counts.div(counts.sum(axis=1), axis=0).mul(100)
    ax = counts.plot(kind="bar", stacked=True, figsize=(max(8, len(counts) * 1.2), 6),
                     color=["tab:blue", "tab:orange"])
    for cont, c in zip(ax.containers, counts.columns):
        ax.bar_label(cont, label_type="center", fontsize=12,
                     labels=[f"{n}\n({p:.1f}%)" if n > 0 else "" for n, p in zip(counts[c], pct[c])])
    ax.set(title=f"{title} and Depression", xlabel=title, ylabel="Number of Students")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Depression")
    ax.grid(axis="y", alpha=0.8)
    save(f"plot_{name}_vs_Depression")


for c in ["Family_background", "Relationship_status", "Marital_status",
          "Living_style", "Health_condition", "friend_circle"]:
    plot_depression_comparison(c)


### ============================================================
### ASSOCIAZIONI CON LA DEPRESSIONE (VARIABILI CATEGORICHE)
### ============================================================
# Tasso di depressione per ogni sottocategoria
rates = pd.concat([
    (df.groupby(c)["dep"].mean() * 100).rename("Depression_Rate").rename_axis("Category")
      .reset_index().assign(Variable=c)
    for c in categorical_columns
])[["Variable", "Category", "Depression_Rate"]].sort_values("Depression_Rate", ascending=False)
log_table("STATISTICHE SULLA DEPRESSIONE IN BASE ALLE SOTTOCATEGORIE", rates)

# Chi-quadro e V di Cramér (correzione di Bonferroni sul numero di variabili)
rows = []
for c in categorical_columns:
    chi2, p, dof, v = assoc(pd.crosstab(df[c], df[TARGET]))
    strength = ("N/A" if np.isnan(v) else "Negligible" if v < 0.1 else
                "Weak" if v < 0.3 else "Moderate" if v < 0.5 else "Strong")
    rows.append({"Variable": c, "Chi-square": chi2, "DoF": dof, "P-value": p,
                 "Cramer's V": v, "Strength": strength})
t = pd.DataFrame(rows).sort_values("Cramer's V", ascending=False)
t["P-value corrected"] = (t["P-value"] * len(categorical_columns)).clip(upper=1)
log_table("ASSOCIAZIONE VARIABILI CATEGORICHE E DEPRESSIONE",
          fmt_p_cols(t, ["P-value", "P-value corrected"]), floatfmt=".3f")

# Condizione di salute
health = pd.crosstab(df["Health_condition"], df[TARGET], margins=True)
health["Tasso_depressione (%)"] = health["Yes"] / health["All"] * 100
log_table("CONDIZIONE DI SALUTE VS DEPRESSIONE", health, index=True)

# Stato di relazione x cerchia di amici (heatmap)
rate = rate_ct("Relationship_status", "friend_circle")
n = pd.crosstab(df["Relationship_status"], df["friend_circle"])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(rate, annot=rate.round(1).astype(str) + "%\n(n=" + n.astype(str) + ")", fmt="",
            cmap="Oranges", cbar_kws={"label": "Tasso di depressione (%)"}, ax=ax)
ax.set(title="Depression by Relationship Status and Friend Circle",
       xlabel="Friend Circle", ylabel="Relationship Status")
save("plot_Relationship_FriendCircle")

# Effetto della cerchia di amici dentro ogni stato di relazione
groups = df["Relationship_status"].dropna().unique()
rows = []
for g in groups:
    s = df[df["Relationship_status"] == g]
    r_yes = s.loc[s["friend_circle"] == "Yes", "dep"].mean() * 100
    r_no = s.loc[s["friend_circle"] == "No", "dep"].mean() * 100
    tab = pd.crosstab(s["friend_circle"], s[TARGET])
    p = chi2_contingency(tab)[1] if min(tab.shape) > 1 else np.nan
    rows.append({"Relationship_status": g, "Students": len(s), "Rate (friends)": r_yes,
                 "Rate (no friends)": r_no, "Difference": r_no - r_yes, "P-value": p,
                 "P-value (corrected)": np.minimum(p * len(groups), 1)})
t = pd.DataFrame(rows).sort_values("Students", ascending=False)
log_table("STATO DI RELAZIONE E CERCHIA DI AMICI VS DEPRESSIONE",
          fmt_p_cols(t, ["P-value", "P-value (corrected)"]), floatfmt=".1f")

# Genere x stato di relazione
log_table("TASSO DI DEPRESSIONE GENERE VS REL STATUS (%)", rate_ct("Relationship_status", "Gender").round(1), index=True)
log_table("N STUDENTI GENERE VS REL STATUS", pd.crosstab(df["Relationship_status"], df["Gender"]), index=True)

# Heatmap V di Cramér tra tutte le variabili categoriche
heat_cols = categorical_columns + [TARGET]
cramers = pd.DataFrame([[assoc(pd.crosstab(df[a], df[b]))[3] for b in heat_cols] for a in heat_cols],
                       index=heat_cols, columns=heat_cols)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cramers, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, ax=ax)
ax.set_title("V di Cramér tra le variabili categoriche")
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
save("plot_vCramer")


### ============================================================
### QUINTILI DI COSTO EXTRA-STUDIO E DI CGPA
### ============================================================
def quintile_analysis(col, qcol, prefix, label, fname):
    """Tasso di depressione per quintili: tabella, grafico, test uno-contro-resto e test complessivo."""
    labels = [f"{prefix}1 (più basso)", f"{prefix}2", f"{prefix}3", f"{prefix}4", f"{prefix}5 (più alto)"]
    df[qcol] = pd.qcut(df[col], q=5, labels=labels)

    res = (df.groupby(qcol, observed=True)
             .agg(Min=(col, "min"), Max=(col, "max"), Students=("dep", "size"),
                  Rate=("dep", lambda x: x.mean() * 100))
             .reset_index())
    log_table(f"TASSO DI DEPRESSIONE PER QUINTILI: {label.upper()}", res)

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(data=res, x=qcol, y="Rate", color="tab:orange")
    ax.axhline(OVERALL, color="gray", linestyle="--", label=f"Tasso complessivo ({OVERALL:.1f}%)")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    ax.set(title=f"Tasso di depressione per quintili: {label}", xlabel=f"{label} (quintili)",
           ylabel="Percentuale di studenti con depressione")
    ax.legend()
    save(fname)

    rows = []
    for q in labels:
        m = df[qcol] == q
        p = chi2_contingency(pd.crosstab(m, df[TARGET]))[1]
        rows.append({"Quintile": q, "Rate (%)": df.loc[m, "dep"].mean() * 100,
                     "Rate others (%)": df.loc[~m, "dep"].mean() * 100,
                     "P-value": p, "P-value corr.": min(p * 5, 1)})
    log_table(f"TEST QUINTILI VS RESTO: {label.upper()}",
              fmt_p_cols(pd.DataFrame(rows), ["P-value", "P-value corr."]), floatfmt=".1f")
    chi2, p, dof, v = assoc(pd.crosstab(df[qcol], df[TARGET]))
    log(f"Test complessivo {label}: chi2={chi2:.2f}, dof={dof}, p-value={fmt_p(p)}, V di Cramér={v:.3f}")


quintile_analysis("Cost_without_study", "Cost_quintile", "Q", "Costo extra-studio", "plot_extra_studio_vs_depressione")
quintile_analysis("CGPA", "CGPA_quintile", "C", "CGPA", "plot_cgpa_quintili_depressione")

# Costo extra-studio x classe sociale ("!" = meno di 20 studenti)
counts = pd.crosstab(df["Cost_quintile"], df["Family_background"])
cls_rates = rate_ct("Cost_quintile", "Family_background").round(1).astype(str) + "%"
cls_rates = cls_rates.mask(counts < 20, cls_rates + "!")
log_table("TASSO DI DEPRESSIONE (%) PER COSTO E CLASSE SOCIALE", cls_rates, index=True, stralign="right")
log_table("N STUDENTI PER COSTO E CLASSE SOCIALE", counts, index=True)

middle = df[df["Family_background"] == "Middle class"]
chi2, p, dof, v = assoc(pd.crosstab(middle["Cost_quintile"], middle[TARGET]))
log(f"\n### CLASSE MEDIA: chi2={chi2:.2f}, dof={dof}, p-value={p:.5f}, V di Cramér={v:.3f}")

rich = df[df["Family_background"] == "Rich class"]
q5 = rich["Cost_quintile"] == "Q5 (più alto)"
log(f"### CLASSE RICCA: Q5={rich.loc[q5, 'dep'].mean() * 100:.1f}% (n={q5.sum()}), "
    f"Q1-Q4={rich.loc[~q5, 'dep'].mean() * 100:.1f}% (n={(~q5).sum()})")


### ============================================================
### ASSOCIAZIONI CON LA DEPRESSIONE (VARIABILI NUMERICHE)
### ============================================================
rows = []
for c in NUM:
    no, yes = df.loc[df[TARGET] == "No", c].dropna(), df.loc[df[TARGET] == "Yes", c].dropna()
    u, p = mannwhitneyu(no, yes, alternative="two-sided")
    rows.append({"Variable": c, "Median (No)": no.median(), "Median (Yes)": yes.median(),
                 "U Statistic": u, "P-value": p, "Effect size (A)": u / (len(no) * len(yes))})
t = pd.DataFrame(rows)
t["P-value corrected"] = (t["P-value"] * len(NUM)).clip(upper=1)
log_table("ASSOCIAZIONE VARIABILI NUMERICHE CON DEPRESSIONE", fmt_p_cols(t, ["P-value", "P-value corrected"]))

corr = df[NUM].corr(method="spearman").round(2)
log_table("CORRELAZIONE DI SPEARMAN", corr, index=True)
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True,
            linewidths=0.8, annot_kws={"size": 12}, ax=ax)
ax.set_title("Correlazione di Spearman tra le variabili numeriche", pad=20)
ax.tick_params(axis="x", rotation=30, labelsize=11)
ax.tick_params(axis="y", rotation=0, labelsize=11)
save("plot_correlazione_spearman")


### ============================================================
### PROFILI DI STUDENTI (IC di Wilson, almeno 20 studenti)
### ============================================================
prof = (df.groupby(["Relationship_status", "friend_circle", "Family_background", "Living_style"], observed=True)
          .agg(Total_Students=("dep", "size"), Depression_Count=("dep", "sum")).reset_index())
log(f"\n### ANALISI PROFILI STUDENTI\nProfili totali: {len(prof)}")
prof = prof[prof["Total_Students"] >= 20].copy()
log(f"Profili con almeno 20 studenti: {len(prof)}")
prof["Depression_Rate"] = prof["Depression_Count"] / prof["Total_Students"] * 100
lo, hi = proportion_confint(prof["Depression_Count"], prof["Total_Students"], method="wilson")
prof["CI_low"], prof["CI_high"] = lo * 100, hi * 100
log_table("TOP 10 PROFILI PER TASSO DI DEPRESSIONE", prof.sort_values("Depression_Rate", ascending=False).head(10))


### ============================================================
### MODELLO 1: DECISION TREE
### ============================================================
tree_cols = NUM + [c for c in categorical_columns if c not in ["Marital_status", "Health_condition"]]
tree_cat = [c for c in tree_cols if c not in NUM]
X_tr, X_te, y_tr, y_te = split(tree_cols, "DECISION TREE")

X_tr_enc = pd.get_dummies(X_tr, columns=tree_cat, drop_first=True, dtype=float)
X_te_enc = (pd.get_dummies(X_te, columns=tree_cat, drop_first=True, dtype=float)
            .reindex(columns=X_tr_enc.columns, fill_value=0))
log(f"\n### ENCODING FEATURE CATEGORICHE\nTrain: {X_tr_enc.shape} | Test: {X_te_enc.shape}\n{X_tr_enc.columns.tolist()}")

baseline = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, class_weight="balanced",
                                  random_state=SEED).fit(X_tr_enc, y_tr)
evaluate("Baseline Tree", y_te, baseline.predict_proba(X_te_enc)[:, 1], store=False)


def search_tree():
    grid = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
            "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 5, 10, 20, 30],
            "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02]}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                      grid, scoring="accuracy", cv=cv, n_jobs=-1)
    with tqdm_joblib(total=int(np.prod([len(v) for v in grid.values()])) * cv.n_splits, desc="GridSearch Tree"):
        gs.fit(X_tr_enc, y_tr)
    log(f"Miglior CV score (Accuracy): {gs.best_score_:.3f}")
    return gs.best_params_


tree_params = load_or_search(PAR_DIR + "gridSearch.par", search_tree)
best_tree = DecisionTreeClassifier(**tree_params, class_weight="balanced", random_state=SEED).fit(X_tr_enc, y_tr)
evaluate("Decision Tree", y_te, best_tree.predict_proba(X_te_enc)[:, 1], 0.5, tree_params)

plt.figure(figsize=(20, 10))
plot_tree(best_tree, feature_names=X_tr_enc.columns, class_names=["No Depression", "Depression"],
          filled=True, rounded=True, max_depth=3, fontsize=9)
plt.title("Albero di classificazione — primi 3 livelli")
save("plot_decision_tree", svg=True)

plot_importance("Decision Tree", X_tr_enc.columns, best_tree.feature_importances_, "Greens_r", "plot_feature_importance")


### ============================================================
### MODELLO 2: RANDOM FOREST (soglia di decisione + analisi OOB)
### ============================================================
RF_CAT = ["Gender", "Family_background", "Relationship_status", "Marital_status",
          "Living_style", "friend_circle", "Home_division"]
X_tr, X_te, y_tr, y_te = split(NUM + RF_CAT, "RANDOM FOREST")

prep = ColumnTransformer([
    ("num", StandardScaler(), NUM),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), RF_CAT),
])
rf_pipe = Pipeline([
    ("preprocessor", prep),
    ("classifier", RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1, class_weight="balanced")),
])


def search_rf():
    dist = {"classifier__max_depth": [4, 6, 8, 10, 15, None],
            "classifier__min_samples_split": randint(2, 21),
            "classifier__min_samples_leaf": randint(1, 21),
            "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
            "classifier__class_weight": ["balanced", "balanced_subsample"]}
    cv, n_iter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED), 60
    rs = RandomizedSearchCV(rf_pipe, dist, n_iter=n_iter, scoring="f1", cv=cv, n_jobs=-1,
                            random_state=SEED, return_train_score=True)
    with tqdm_joblib(total=n_iter * cv.n_splits, desc="RandomSearch RF"):
        rs.fit(X_tr, y_tr)
    log(f"Miglior F1 score in CV: {rs.best_score_:.3f}")
    top = (pd.DataFrame(rs.cv_results_).sort_values("rank_test_score").head(10)
           [["rank_test_score", "mean_test_score", "std_test_score", "mean_train_score", "params"]])
    log_table("TOP 10 COMBINAZIONI CV RANDOM FOREST", top, floatfmt=".4f")
    return rs.best_params_


rf_params = {k.replace("classifier__", ""): v
             for k, v in load_or_search(PAR_DIR + "randomSearchRandomForest.par", search_rf).items()}

# Curva OOB al crescere del numero di alberi
X_tr_t = prep.fit_transform(X_tr)
oob_rf = RandomForestClassifier(**rf_params, n_estimators=50, oob_score=True, warm_start=True,
                                random_state=SEED, n_jobs=-1)
rows = []
for n_est in range(50, 601, 25):
    oob_rf.set_params(n_estimators=n_est).fit(X_tr_t, y_tr)
    rows.append({"n_estimators": n_est, "OOB_error": 1 - oob_rf.oob_score_,
                 "OOB_ROC_AUC": roc_auc_score(y_tr, oob_rf.oob_decision_function_[:, 1])})
oob = pd.DataFrame(rows)
best_row = oob.loc[oob["OOB_ROC_AUC"].idxmax()]
best_n = int(best_row["n_estimators"])
log(f"\n### ANALISI OOB PER NUMERO DI ALBERI\nAlberi con OOB ROC-AUC massimo: {best_n} | "
    f"OOB ROC-AUC: {best_row['OOB_ROC_AUC']:.4f} | OOB error: {best_row['OOB_error']:.4f}")

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(oob["n_estimators"], oob["OOB_error"], marker="o", color="tab:red")
ax1.set_xlabel("Numero di alberi")
ax1.set_ylabel("OOB error", color="tab:red")
ax2 = ax1.twinx()
ax2.plot(oob["n_estimators"], oob["OOB_ROC_AUC"], marker="o", color="tab:green")
ax2.set_ylabel("OOB ROC-AUC", color="tab:green")
ax2.grid(False)
ax1.axvline(best_n, color="black", linestyle="--", label=f"Scelta: {best_n} alberi")
ax1.legend()
fig.suptitle("Andamento OOB in funzione del numero di alberi")
save("plot_rf_oob_curve", svg=True)

# Modello finale
final_n = max(best_n, 300)
rf_pipe.set_params(classifier=RandomForestClassifier(**rf_params, n_estimators=final_n, oob_score=True,
                                                     random_state=SEED, n_jobs=-1))
rf_pipe.fit(X_tr, y_tr)
rf = rf_pipe.named_steps["classifier"]
log(f"\n### VALUTAZIONE OOB MODELLO FINALE RF\nAlberi: {final_n} | OOB accuracy: {rf.oob_score_:.4f} | "
    f"OOB error: {1 - rf.oob_score_:.4f} | OOB ROC-AUC: {roc_auc_score(y_tr, rf.oob_decision_function_[:, 1]):.4f}")

prob = rf_pipe.predict_proba(X_te)[:, 1]
evaluate("Random Forest (soglia 0.50)", y_te, prob, 0.5, store=False)

# Soglia che massimizza l'F1 (calcolata sul test set)
p_, r_, th_ = precision_recall_curve(y_te, prob)
thr_tab = pd.DataFrame({"threshold": th_, "precision": p_[:-1], "recall": r_[:-1]})
thr_tab["f1"] = 2 * thr_tab["precision"] * thr_tab["recall"] / (thr_tab["precision"] + thr_tab["recall"] + 1e-12)
best_idx = thr_tab["f1"].idxmax()
rf_thr = float(thr_tab.loc[best_idx, "threshold"])
log_table("OTTIMIZZAZIONE SOGLIA DI DECISIONE RF", thr_tab.loc[[best_idx]], floatfmt=".4f")

plt.figure(figsize=(10, 5))
for c, color, lw in [("precision", "tab:blue", 1.5), ("recall", "tab:orange", 1.5), ("f1", "tab:green", 2)]:
    plt.plot(thr_tab["threshold"], thr_tab[c], label=c.capitalize(), color=color, linewidth=lw)
plt.axvline(rf_thr, color="black", linestyle="--", label=f"Soglia ottimale = {rf_thr:.3f}")
plt.xlabel("Soglia di probabilità")
plt.ylabel("Metrica")
plt.ylim(0, 1.05)
plt.title("Precision, Recall e F1 in funzione della soglia")
plt.legend()
save("plot_rf_threshold_tuning", svg=True)

evaluate("Random Forest", y_te, prob, rf_thr, rf_params)
plot_importance("Random Forest", prep.get_feature_names_out(), rf.feature_importances_, "Blues_r", "plot_rf_feature_importance")


### ============================================================
### MODELLO 3: CATBOOST
### ============================================================
df["male"] = (df["Gender"] == "Male").astype(int)
df["no_friends"] = (df["friend_circle"] == "No").astype(int)

CB_CAT = ["Relationship_status", "Family_background", "Home_division", "Living_style"]
CB_NUM = ["CGPA", "Cost_without_study", "Age", "Current_semester"]
X_tr, X_te, y_tr, y_te = split(["male", "no_friends"] + CB_CAT + CB_NUM, "CATBOOST")


def new_cb(depth, l2_leaf_reg, learning_rate=0.03, iterations=300):
    return CatBoostClassifier(
        iterations=iterations, depth=depth, learning_rate=learning_rate, l2_leaf_reg=l2_leaf_reg,
        one_hot_max_size=10, cat_features=tuple(CB_CAT), auto_class_weights="Balanced",
        random_seed=SEED, verbose=0, allow_writing_files=False,
    )


def search_cb():
    cv_rep = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows = []
    for depth, l2 in product([4, 6, 8], [10, 30, 50]):
        m = new_cb(depth, l2)
        scores = cross_val_score(m, X_tr, y_tr, cv=cv_rep, scoring="roc_auc")
        m.fit(X_tr, y_tr)
        auc_train = roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])
        rows.append([depth, l2, scores.mean(), scores.std(), auc_train, auc_train - scores.mean()])
    grid = (pd.DataFrame(rows, columns=["depth", "l2_leaf_reg", "AUC cv", "std cv", "AUC train", "scarto train-cv"])
            .sort_values("AUC cv", ascending=False).reset_index(drop=True))
    log_table("RISULTATI GRID SEARCH CATBOOST", grid, floatfmt=".4f")
    return {"depth": int(grid.loc[0, "depth"]), "l2_leaf_reg": float(grid.loc[0, "l2_leaf_reg"]),
            "learning_rate": 0.03}


cb_params = load_or_search(PAR_DIR + "catboost.par", search_cb)
model = new_cb(**cb_params).fit(X_tr, y_tr)
plot_importance("CatBoost", X_tr.columns, model.get_feature_importance(), "Oranges_r", "plot_catboost_feature_importance")

# Quanti alberi servono? (AUC train vs cross-validation)
N_MAX = 1000
steps = list(range(25, N_MAX + 1, 25))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
curve_val, curve_tr = [], []
for i_tr, i_va in cv.split(X_tr, y_tr):
    m = new_cb(**cb_params, iterations=N_MAX).fit(X_tr.iloc[i_tr], y_tr.iloc[i_tr])
    for store_, idx in ((curve_tr, i_tr), (curve_val, i_va)):
        store_.append([roc_auc_score(y_tr.iloc[idx], m.predict_proba(X_tr.iloc[idx], ntree_end=k)[:, 1])
                       for k in steps])
curve = pd.DataFrame({"alberi": steps, "AUC cv": np.mean(curve_val, axis=0), "AUC train": np.mean(curve_tr, axis=0)})

plt.figure(figsize=(7, 4.5))
plt.plot(curve["alberi"], curve["AUC train"], label="train")
plt.plot(curve["alberi"], curve["AUC cv"], label="cross-validation")
plt.axvline(300, color="k", ls="--", label="300 alberi (usati)")
plt.xlabel("numero di alberi")
plt.ylabel("AUC")
plt.title("Quanti alberi servono?")
plt.legend()
save("plot_catboost_curva_alberi", svg=True)
log_table("OTTIMIZZAZIONE NUMERO ALBERI CATBOOST", curve.loc[[curve["AUC cv"].idxmax()]], floatfmt=".4f")

# Soglia di decisione ottimale (F1 massimo) in cross-validation sul train
p_cv = cross_val_predict(new_cb(**cb_params), X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
rows = []
for t_ in np.arange(0.05, 0.96, 0.01):
    pred = (p_cv >= t_).astype(int)
    rows.append([t_, recall_score(y_tr, pred), precision_score(y_tr, pred, zero_division=0),
                 f1_score(y_tr, pred), pred.mean()])
thr_cb = pd.DataFrame(rows, columns=["soglia", "recall", "precision", "f-score", "% segnalati"])
log_table("ANALISI SOGLIE DI DECISIONE CATBOOST (CV)",
          thr_cb[(thr_cb["soglia"] * 100).round().astype(int) % 5 == 0], floatfmt=".4f")
cb_thr = round(float(thr_cb.loc[thr_cb["f-score"].idxmax(), "soglia"]), 2)
log(f"\n### SOGLIA OTTIMALE CATBOOST: {cb_thr}")

# Valutazione finale (train vs test)
prob_tr, prob_te = model.predict_proba(X_tr)[:, 1], model.predict_proba(X_te)[:, 1]
evaluate("CatBoost", y_te, prob_te, cb_thr, cb_params)

metric_fns = {"accuracy": accuracy_score, "precision": precision_score, "recall": recall_score,
              "f-score": f1_score, "phi-score": matthews_corrcoef}
tr_te = pd.DataFrame({
    name: {**{k: f(y, (p >= cb_thr).astype(int)) for k, f in metric_fns.items()}, "AUC": roc_auc_score(y, p)}
    for name, (y, p) in {"train": (y_tr, prob_tr), "test": (y_te, prob_te)}.items()
})
log_table("METRICHE TRAIN VS TEST CATBOOST", tr_te, floatfmt=".4f", index=True)
log(f"Accuracy 'rispondo sempre No' (baseline): {1 - y_te.mean():.3f}")


### ============================================================
### MODELLO 4: REGRESSIONE LOGISTICA (forward/backward con AIC/BIC)
### ============================================================
X_tr, X_te, y_tr, y_te = split(NUM + RF_CAT, "REGRESSIONE LOGISTICA")

# Standardizzazione (fit solo sul train) + one-hot + intercetta
X_tr, X_te = X_tr.copy(), X_te.copy()
scaler = StandardScaler()
X_tr[NUM] = scaler.fit_transform(X_tr[NUM])
X_te[NUM] = scaler.transform(X_te[NUM])
X_tr_lg = sm.add_constant(pd.get_dummies(X_tr, columns=RF_CAT, drop_first=True, dtype=float), has_constant="add")
X_te_lg = sm.add_constant(pd.get_dummies(X_te, columns=RF_CAT, drop_first=True, dtype=float), has_constant="add")
X_te_lg = X_te_lg.reindex(columns=X_tr_lg.columns, fill_value=0)
log(f"\n### ENCODING REGRESSIONE LOGISTICA\nTrain: {X_tr_lg.shape} | Test: {X_te_lg.shape}\n{X_tr_lg.columns.tolist()}")


def fit_logit(X, y):
    """Logit con fallback sui solver; scarta colonne costanti (tranne 'const'). None se non converge."""
    X = X.loc[:, ~X.columns.duplicated()].apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X[[c for c in X.columns if c == "const" or X[c].nunique(dropna=False) > 1]]
    y = pd.to_numeric(y, errors="coerce")
    valid = X.notna().all(axis=1) & y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    if y.nunique() < 2 or X.shape[1] == 0:
        return None
    for solver in ["newton", "lbfgs", "bfgs"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(y, X).fit(method=solver, maxiter=500, disp=0)
            if m.mle_retvals.get("converged", False):
                return m
        except Exception:
            continue
    return None


def criterion(model, crit):
    """AIC o BIC del modello; infinito se non valido."""
    return np.inf if model is None else float(getattr(model, crit))


def stepwise(direction, crit):
    """Forward selection o backward elimination guidata da AIC/BIC -> lista di variabili."""
    predictors = [c for c in X_tr_lg.columns if c != "const"]
    selected = [] if direction == "forward" else predictors.copy()
    remaining = predictors.copy() if direction == "forward" else []
    fit = lambda cols: fit_logit(X_tr_lg[["const"] + cols], y_tr)
    current = criterion(fit(selected), crit)
    while (remaining if direction == "forward" else len(selected) > 1):
        pool = remaining if direction == "forward" else selected
        trials = [(criterion(fit(selected + [c] if direction == "forward" else [v for v in selected if v != c]), crit), c)
                  for c in pool]
        score, cand = min(trials)
        if score >= current - 1e-6:
            break
        if direction == "forward":
            selected.append(cand), remaining.remove(cand)
        else:
            selected.remove(cand)
        current = score
        log(f"[{direction.upper()}] {'+' if direction == 'forward' else '-'} {cand:<40} {crit.upper()} = {score:.3f}")
    return selected


def search_logit():
    """Le 4 combinazioni metodo x criterio -> {chiave: variabili selezionate}."""
    out = {}
    for direction, crit in product(["forward", "backward"], ["aic", "bic"]):
        log(f"\n### {direction.upper()} SELECTION — {crit.upper()}")
        out[f"{direction}_{crit}"] = stepwise(direction, crit)
    return out


logit_vars = load_or_search(PAR_DIR + "logit.par", search_logit)
# se il .par esistente ha un formato diverso da {chiave: [variabili]}, rifà la selezione e lo sovrascrive
if not (isinstance(logit_vars, dict) and logit_vars and all(
        isinstance(v, list) and set(v) <= set(X_tr_lg.columns) for v in logit_vars.values())):
    log("Formato di logit.par non compatibile con le variabili attuali: rifaccio la selezione.")
    logit_vars = search_logit()
    joblib.dump(logit_vars, PAR_DIR + "logit.par")
logit_models = {k: fit_logit(X_tr_lg[["const"] + v], y_tr) for k, v in logit_vars.items()}

FINAL_CRITERION = "bic"  # "aic" privilegia l'adattamento, "bic" la parsimonia
valid = {k: m for k, m in logit_models.items() if m is not None}
if not valid:
    raise RuntimeError("Nessun modello logit è stato stimato con successo.")

log_table("RIEPILOGO MODELLI LOGIT", pd.DataFrame([{
    "Metodo": k.replace("_", " ").title(), "N. variabili": len(logit_vars[k]),
    "AIC train": np.nan if m is None else m.aic, "BIC train": np.nan if m is None else m.bic,
    "Variabili selezionate": ", ".join(logit_vars[k]) or "Nessuna"} for k, m in logit_models.items()]))

best_key = min(valid, key=lambda k: criterion(valid[k], FINAL_CRITERION))
logit_model = valid[best_key]
log(f"\n### MODELLO LOGIT SELEZIONATO: {best_key} (criterio {FINAL_CRITERION.upper()}, "
    f"{len(logit_vars[best_key])} variabili)\n{logit_model.summary()}")

# Test set: stesse colonne usate dal modello (quelle costanti nel train sono state scartate)
logit_prob = logit_model.predict(X_te_lg[logit_model.params.index].astype(float))
evaluate("Logistic Regression", y_te, logit_prob, 0.5,
         {"selezione": best_key, "criterio": FINAL_CRITERION.upper(), "n_variabili": len(logit_vars[best_key])})

or_tab = pd.DataFrame({"Coefficiente": logit_model.params, "Odds Ratio": np.exp(logit_model.params),
                       "p-value": logit_model.pvalues}).sort_values("p-value")
log_table("ODDS RATIO REGRESSIONE LOGISTICA", or_tab, floatfmt=".4f", index=True)


### ============================================================
### CONFRONTO FINALE TRA I MODELLI
### ============================================================
COLORS = {"Decision Tree": "tab:green", "Random Forest": "tab:blue", "CatBoost": "tab:orange",
          "Logistic Regression": "tab:red"}

metrics_df = pd.DataFrame({
    name: {"Accuracy": accuracy_score(r["y_true"], r["y_pred"]),
           "Precision": precision_score(r["y_true"], r["y_pred"], zero_division=0),
           "Recall": recall_score(r["y_true"], r["y_pred"]),
           "F1": f1_score(r["y_true"], r["y_pred"]),
           "ROC-AUC": roc_auc_score(r["y_true"], r["y_prob"])}
    for name, r in model_results.items()
}).T
params_df = pd.DataFrame({
    "Modello": list(model_results),
    "Soglia": [r["threshold"] for r in model_results.values()],
    "Iperparametri": [str(r["params"]) for r in model_results.values()],
})
log_table("CONFRONTO MODELLI - METRICHE (TEST SET)", metrics_df, floatfmt=".4f", index=True)
log_table("CONFRONTO MODELLI - PARAMETRI E SOGLIE", params_df, floatfmt=".3f")

# Plot 1: curve ROC a confronto
plt.figure(figsize=(8, 7))
for name, r in model_results.items():
    fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
    plt.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={roc_auc_score(r['y_true'], r['y_prob']):.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Caso")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("Curve ROC a confronto (test set)")
plt.legend(loc="lower right")
save("plot_confronto_roc", svg=True, dpi=150)

# Plot 2: metriche sul test set a confronto
ax = metrics_df.T.plot(kind="bar", figsize=(11, 6), color=[COLORS[m] for m in metrics_df.index], width=0.8)
for cont in ax.containers:
    ax.bar_label(cont, fmt="%.2f", fontsize=8, padding=2)
ax.set(title="Metriche sul test set", ylabel="Valore", ylim=(0, 1.1))
ax.tick_params(axis="x", rotation=0)
ax.legend(title="Modello")
save("plot_confronto_metriche", svg=True, dpi=150)

log(f"\n### Grafici di confronto salvati in {PLOTS}")
