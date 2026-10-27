# ============================================================
# 1. IMPORTAZIONI
# ============================================================

import argparse
import json

from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import plotly.express as px


# ============================================================
# 2. CONFIGURAZIONE DEI PERCORSI
# ============================================================

# Cartella principale del progetto.
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Cartella in cui salvare tutti i risultati.
OUTPUT_DIR = PROJECT_DIR / "analisi"
PLOTS_DIR = OUTPUT_DIR / "plots"


# ============================================================
# 3. CARICAMENTO DEL DATASET
# ============================================================

def load_dataset(file_path):
    """Carica un dataset CSV o Parquet."""

    file_path = Path(file_path)

    if not file_path.is_absolute():
        file_path = PROJECT_DIR / file_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset non trovato: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(file_path)

    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(file_path)

    raise ValueError(
        "Formato non supportato. Usa CSV o Parquet."
    )


# ============================================================
# 4. CREAZIONE DELLE CARTELLE
# ============================================================

def prepare_output_dirs():
    """Crea le cartelle necessarie."""

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 5. SALVATAGGIO DEI GRAFICI
# ============================================================

def save_plot(fig, plot_name):
    """Salva un grafico in HTML e JSON."""

    html_path = PLOTS_DIR / f"{plot_name}.html"
    json_path = PLOTS_DIR / f"{plot_name}.json"

    # HTML autonomo, consultabile nel browser.
    fig.write_html(
        str(html_path),
        include_plotlyjs=True,
        full_html=True
    )

    # JSON riutilizzabile direttamente in Streamlit.
    json_path.write_text(
        fig.to_json(),
        encoding="utf-8"
    )


# ============================================================
# 6. STATISTICHE DESCRITTIVE
# ============================================================

def save_statistics(df):
    """Salva le statistiche e le informazioni sul dataset."""

    # Statistiche delle colonne numeriche.
    numeric_df = df.select_dtypes(include="number")

    numeric_stats = numeric_df.describe().T

    numeric_stats.to_csv(
        OUTPUT_DIR / "statistiche_descrittive.csv"
    )

    # Informazioni sulla qualità delle colonne.
    column_info = pd.DataFrame({
        "colonna": df.columns,
        "tipo": df.dtypes.astype(str).values,
        "valori_mancanti": df.isna().sum().values,
        "percentuale_mancanti": (
            df.isna().mean().values * 100
        ),
        "valori_unici": df.nunique().values
    })

    column_info.to_csv(
        OUTPUT_DIR / "informazioni_colonne.csv",
        index=False
    )

    # Conteggio delle righe duplicate.
    duplicate_rows = int(df.duplicated().sum())

    # Riepilogo generale in JSON.
    info = {
        "dataset": str(df.shape),
        "numero_righe": int(df.shape[0]),
        "numero_colonne": int(df.shape[1]),
        "colonne_numeriche": int(len(numeric_df.columns)),
        "valori_mancanti": int(df.isna().sum().sum()),
        "righe_duplicate": duplicate_rows,
        "data_generazione_utc": datetime.now(
            timezone.utc
        ).isoformat()
    }

    (OUTPUT_DIR / "informazioni_dataset.json").write_text(
        json.dumps(
            info,
            indent=4,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    # Statistiche delle colonne categoriche.
    categorical_df = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    )

    if not categorical_df.empty:

        categorical_stats = pd.DataFrame({
            "colonna": categorical_df.columns,
            "valori_unici": [
                categorical_df[col].nunique()
                for col in categorical_df.columns
            ],
            "valore_piu_frequente": [
                (
                    categorical_df[col].mode().iloc[0]
                    if not categorical_df[col].mode().empty
                    else None
                )
                for col in categorical_df.columns
            ]
        })

        categorical_stats.to_csv(
            OUTPUT_DIR / "statistiche_categoriche.csv",
            index=False
        )

    return info, column_info


# ============================================================
# 7. GENERAZIONE DEGLI ISTOGRAMMI
# ============================================================

def generate_numeric_plots(df):
    """Genera un istogramma per ogni colonna numerica."""

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    for column in numeric_columns:

        # Esclude i valori mancanti.
        values = df[column].dropna()

        if values.empty:
            continue

        fig = px.histogram(
            df,
            x=column,
            nbins=30,
            title=f"Distribuzione di {column}",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequenza"
        )

        save_plot(
            fig,
            f"distribuzione_{column}"
        )


# ============================================================
# 8. GENERAZIONE DEI GRAFICI CATEGORICI
# ============================================================

def generate_categorical_plots(df):
    """Genera grafici a barre per le colonne categoriche."""

    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    for column in categorical_columns:

        # Mostra le 20 categorie più frequenti.
        counts = (
            df[column]
            .astype("string")
            .fillna("Valore mancante")
            .value_counts()
            .head(20)
            .rename_axis(column)
            .reset_index(name="frequenza")
        )

        if counts.empty:
            continue

        fig = px.bar(
            counts,
            x=column,
            y="frequenza",
            title=f"Frequenza delle categorie: {column}",
            template="plotly_white"
        )

        fig.update_layout(
            xaxis_title=column,
            yaxis_title="Frequenza"
        )

        save_plot(
            fig,
            f"frequenze_{column}"
        )


# ============================================================
# 9. GRAFICO DEI VALORI MANCANTI
# ============================================================

def generate_missing_plot(df):
    """Genera un grafico dei valori mancanti."""

    missing = (
        df.isna()
        .sum()
        .rename("valori_mancanti")
        .rename_axis("colonna")
        .reset_index()
    )

    missing = missing[
        missing["valori_mancanti"] > 0
    ]

    if missing.empty:
        return

    fig = px.bar(
        missing,
        x="colonna",
        y="valori_mancanti",
        title="Valori mancanti per colonna",
        template="plotly_white"
    )

    save_plot(
        fig,
        "valori_mancanti"
    )


# ============================================================
# 10. GENERAZIONE DEL REPORT MARKDOWN
# ============================================================

def generate_markdown_report(df, info, column_info):
    """Crea un report testuale con le informazioni principali."""

    report = [
        "# Report di analisi esplorativa",
        "",
        "## Informazioni generali",
        "",
        f"- Righe: {info['numero_righe']}",
        f"- Colonne: {info['numero_colonne']}",
        f"- Valori mancanti: {info['valori_mancanti']}",
        f"- Righe duplicate: {info['righe_duplicate']}",
        f"- Colonne numeriche: {info['colonne_numeriche']}",
        "",
        "## Informazioni sulle colonne",
        "",
        column_info.to_markdown(index=False),
        "",
        "## Statistiche descrittive",
        ""
    ]

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:

        report.extend([
            numeric_df.describe().T.to_markdown(),
            ""
        ])

    else:

        report.extend([
            "Non sono presenti colonne numeriche.",
            ""
        ])

    report.extend([
        "## Grafici generati",
        "",
        "I grafici interattivi sono disponibili nella cartella `plots/`.",
        ""
    ])

    (OUTPUT_DIR / "report.md").write_text(
        "\n".join(report),
        encoding="utf-8"
    )


# ============================================================
# 11. ESECUZIONE COMPLETA
# ============================================================

def main():
    """Esegue l'intera pipeline di generazione del report."""

    parser = argparse.ArgumentParser(
        description="Genera grafici e statistiche esplorative."
    )

    parser.add_argument(
        "dataset",
        help="Percorso del file CSV o Parquet."
    )

    args = parser.parse_args()

    prepare_output_dirs()

    # Carica il dataset.
    df = load_dataset(args.dataset)

    if df.empty or len(df.columns) == 0:
        raise ValueError(
            "Il dataset è vuoto oppure non contiene colonne."
        )

    print("Dataset caricato:", df.shape)

    # Genera statistiche e grafici.
    info, column_info = save_statistics(df)

    generate_numeric_plots(df)
    generate_categorical_plots(df)
    generate_missing_plot(df)

    generate_markdown_report(
        df,
        info,
        column_info
    )

    print("Report generato correttamente.")
    print("Risultati salvati in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
