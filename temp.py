import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

RANDOM_STATE = 42
TARGET = "Depression"

# Variabili indipendenti
num_cols = [
    "Age",
    "Current_semester",
    "CGPA",
    "Cost_without_study"
]

cat_cols = [
    "Gender",
    "Family_background",
    "Relationship_status",
    "Marital_status",
    "Living_style",
    "friend_circle",
    "Home_division"
]

# Variabile esclusa per possibile data leakage / separazione perfetta
excluded_cols = ["Health_condition"]

print("Variabili numeriche:", num_cols)
print("Variabili categoriche:", cat_cols)
print("Variabili escluse:", excluded_cols)

df_model = df_ready.copy()

# Rimuove eventuali colonne di clustering o non necessarie
cols_to_drop = [
    col for col in df_model.columns
    if col.startswith("cluster")
]
df_model = df_model.drop(columns=cols_to_drop, errors="ignore")

# Mantiene solo le colonne utili
needed_cols = [TARGET] + num_cols + cat_cols
df_model = df_model[needed_cols].copy()

# Rimuove righe incomplete
df_model = df_model.dropna(subset=needed_cols).copy()

# Codifica del target binario
df_model[TARGET] = (
    df_model[TARGET]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": 1, "no": 0})
)

print("Distribuzione del target:")
print(df_model[TARGET].value_counts(dropna=False))

if df_model[TARGET].isna().any():
    raise ValueError(
        "Il target contiene valori non riconosciuti. "
        "Attesi: Yes / No."
    )

df_model[TARGET] = df_model[TARGET].astype(int)

print("\nDimensioni finali:", df_model.shape)

train_df, test_df = train_test_split(
    df_model,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df_model[TARGET]
)

print("Train:", train_df.shape)
print("Test:", test_df.shape)

print("\nDistribuzione target nel train:")
print(train_df[TARGET].value_counts(normalize=True).round(3))

print("\nDistribuzione target nel test:")
print(test_df[TARGET].value_counts(normalize=True).round(3))

# Standardizzazione delle numeriche: fit solo sul training set
scaler = StandardScaler()

train_df[num_cols] = scaler.fit_transform(train_df[num_cols])
test_df[num_cols] = scaler.transform(test_df[num_cols])

# One-hot encoding delle categoriche
train_X = pd.get_dummies(
    train_df[num_cols + cat_cols],
    columns=cat_cols,
    drop_first=True,
    dtype=float
)

test_X = pd.get_dummies(
    test_df[num_cols + cat_cols],
    columns=cat_cols,
    drop_first=True,
    dtype=float
)

# Allinea le colonne del test a quelle del train
test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

# Intercetta per statsmodels
X_train = sm.add_constant(train_X, has_constant="add")
X_test = sm.add_constant(test_X, has_constant="add")

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("\nColonne finali:")
print(X_train.columns.tolist())

def fit_logit(X, y):
    X = X.copy()
    y = y.copy()

    # Rimuove colonne duplicate
    X = X.loc[:, ~X.columns.duplicated()]

    # Forza tipi numerici
    X = X.apply(pd.to_numeric, errors="coerce").astype(float)
    X = X.replace([np.inf, -np.inf], np.nan)

    # Rimuove le colonne costanti MA MANTIENE SEMPRE "const"
    cols_to_keep = [
        col for col in X.columns 
        if col == "const" or X[col].nunique(dropna=False) > 1
    ]
    X = X[cols_to_keep]

    y = pd.to_numeric(y, errors="coerce")

    valid = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    if y.nunique() < 2 or X.shape[1] == 0:
        return None

    methods = ["newton", "lbfgs", "bfgs"]

    for method in methods:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                model = sm.Logit(y, X).fit(
                    method=method,
                    maxiter=500,
                    disp=0
                )

            if model.mle_retvals.get("converged", False):
                return model

        except Exception:
            continue

    return None


def get_criterion(model, criterion_name):
    """Restituisce AIC o BIC; infinito se il modello non è valido."""
    if model is None:
        return np.inf

    criterion_name = criterion_name.lower()

    if criterion_name not in ["aic", "bic"]:
        raise ValueError("criterion_name deve essere 'aic' oppure 'bic'")

    return float(getattr(model, criterion_name))

def forward_selection(
    X_train,
    y_train,
    criterion_name="aic",
    verbose=True
):
    """
    Forward selection basata su AIC oppure BIC.
    """
    predictors = [
        col for col in X_train.columns
        if col != "const"
    ]

    selected = []
    remaining = predictors.copy()

    # Modello iniziale: solo intercetta
    current_model = fit_logit(X_train[["const"]], y_train)
    current_score = get_criterion(current_model, criterion_name)
    best_model = current_model

    while remaining:
        trials = []

        for candidate in remaining:
            cols = ["const"] + selected + [candidate]

            model = fit_logit(X_train[cols], y_train)
            score = get_criterion(model, criterion_name)

            trials.append((score, candidate, model))

        trials.sort(key=lambda x: x[0])

        best_trial_score, best_candidate, best_trial_model = trials[0]

        if best_trial_score < current_score - 1e-6:
            selected.append(best_candidate)
            remaining.remove(best_candidate)

            current_score = best_trial_score
            best_model = best_trial_model

            if verbose:
                print(
                    f"[FORWARD] + {best_candidate:<40} "
                    f"{criterion_name.upper()} = {best_trial_score:.3f}"
                )
        else:
            break

    return selected, best_model, current_score

def backward_elimination(
    X_train,
    y_train,
    criterion_name="aic",
    verbose=True
):
    """
    Backward elimination basata su AIC oppure BIC.
    """
    selected = [
        col for col in X_train.columns
        if col != "const"
    ]

    current_model = fit_logit(
        X_train[["const"] + selected],
        y_train
    )

    current_score = get_criterion(current_model, criterion_name)
    best_model = current_model

    while len(selected) > 1:
        trials = []

        for candidate in selected:
            trial_vars = [
                col for col in selected
                if col != candidate
            ]

            model = fit_logit(
                X_train[["const"] + trial_vars],
                y_train
            )

            score = get_criterion(model, criterion_name)
            trials.append((score, candidate, model))

        trials.sort(key=lambda x: x[0])

        best_trial_score, candidate_to_remove, best_trial_model = trials[0]

        if best_trial_score < current_score - 1e-6:
            selected.remove(candidate_to_remove)

            current_score = best_trial_score
            best_model = best_trial_model

            if verbose:
                print(
                    f"[BACKWARD] - {candidate_to_remove:<40} "
                    f"{criterion_name.upper()} = {best_trial_score:.3f}"
                )
        else:
            break

    return selected, best_model, current_score

results = {}

for method in ["forward", "backward"]:
    for criterion_name in ["aic", "bic"]:
        key = f"{method}_{criterion_name}"

        print(f"\n{'=' * 70}")
        print(f"{method.upper()} SELECTION — {criterion_name.upper()}")
        print(f"{'=' * 70}")

        if method == "forward":
            variables, model, score = forward_selection(
                X_train,
                y_train,
                criterion_name=criterion_name,
                verbose=True
            )
        else:
            variables, model, score = backward_elimination(
                X_train,
                y_train,
                criterion_name=criterion_name,
                verbose=True
            )

        results[key] = {
            "variables": variables,
            "model": model,
            "score": score
        }

        summary_rows = []

for key, result in results.items():
    model = result["model"]

    summary_rows.append({
        "Metodo": key.replace("_", " ").title(),
        "N. variabili": len(result["variables"]),
        "AIC train": np.nan if model is None else round(model.aic, 2),
        "BIC train": np.nan if model is None else round(model.bic, 2),
        "Variabili selezionate": (
            ", ".join(result["variables"])
            if result["variables"]
            else "Nessuna"
        )
    })

summary = pd.DataFrame(summary_rows)

print("\nRiepilogo dei modelli:")
print(summary.to_string(index=False))

# Scegli il criterio per il modello finale:
# "aic" privilegia l'adattamento;
# "bic" privilegia modelli più parsimoniosi.
final_criterion = "bic"

valid_results = {
    key: result
    for key, result in results.items()
    if result["model"] is not None
}

if not valid_results:
    raise RuntimeError(
        "Nessun modello logit è stato stimato con successo."
    )

best_key = min(
    valid_results,
    key=lambda key: get_criterion(
        valid_results[key]["model"],
        final_criterion
    )
)

best_result = valid_results[best_key]
best_vars = best_result["variables"]
best_model = best_result["model"]

print(f"\nModello selezionato: {best_key}")
print(f"Criterio: {final_criterion.upper()}")
print(f"Numero variabili: {len(best_vars)}")
print(f"Variabili finali:\n{best_vars}")

print("\nRiepilogo del modello:")
print(best_model.summary())

best_result = valid_results[best_key]
best_vars = best_result["variables"]

# Ristima robusta del modello migliore sulle sole variabili selezionate
best_model = fit_logit(X_train[["const"] + best_vars], y_train)

if best_model is None:
    raise RuntimeError("Il modello migliore non è riuscito a convergere durante la ristima.")

print(f"\nModello selezionato: {best_key}")
print(f"Criterio: {final_criterion.upper()}")
print(f"Numero variabili: {len(best_vars)}")
print(f"Variabili finali:\n{best_vars}")

print("\nRiepilogo del modello:")
print(best_model.summary())

# Prepara il test set con esattamente le stesse variabili del modello migliore
X_test_best = X_test[["const"] + best_vars].copy()

# Sicurezza: rimuove duplicati e allinea le colonne
X_test_best = X_test_best.loc[:, ~X_test_best.columns.duplicated()]
X_test_best = X_test_best.reindex(
    columns=["const"] + best_vars,
    fill_value=0
).astype(float)

# Probabilità e classi predette
y_prob = best_model.predict(X_test_best)
y_pred = (y_prob >= 0.5).astype(int)

print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))
print("ROC-AUC:", round(roc_auc_score(y_test, y_prob), 3))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Odds ratio
odds_ratios = np.exp(best_model.params)

results_or = pd.DataFrame({
    "Coefficiente": best_model.params,
    "Odds Ratio": odds_ratios,
    "p-value": best_model.pvalues
}).sort_values("p-value")

print(results_or.round(4))