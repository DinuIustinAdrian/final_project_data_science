
# ============================================================
# 1. IMPORTAZIONI
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

from io import BytesIO


# ============================================================
# 2. CONFIGURAZIONE DELLA PAGINA
# ============================================================

st.set_page_config(
    page_title="Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 3. FUNZIONI DI SUPPORTO
# ============================================================

@st.cache_data(show_spinner=False)
def load_dataset(file_bytes, file_name):
    """Carica un dataset CSV o Parquet."""

    if file_name.lower().endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))

    if file_name.lower().endswith(".parquet"):
        return pd.read_parquet(BytesIO(file_bytes))

    raise ValueError("Formato non supportato.")


def get_numeric_columns(df):
    """Restituisce le colonne numeriche."""

    return df.select_dtypes(include="number").columns.tolist()


def get_categorical_columns(df):
    """Restituisce le colonne categoriche."""

    return df.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns.tolist()


# ============================================================
# 4. FUNZIONI PER I GRAFICI
# ============================================================

def plot_histogram(df, column):
    """Crea un istogramma."""

    fig = px.histogram(
        df,
        x=column,
        title=f"Distribuzione di {column}",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequenza"
    )

    return fig


def plot_bar_chart(df, column):
    """Crea un grafico a barre per una variabile categorica."""

    counts = (
        df[column]
        .astype("string")
        .fillna("Valore mancante")
        .value_counts()
        .head(30)
        .rename_axis(column)
        .reset_index(name="Frequenza")
    )

    fig = px.bar(
        counts,
        x=column,
        y="Frequenza",
        title=f"Frequenza di {column}",
        template="plotly_white"
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequenza"
    )

    return fig


def plot_boxplot(df, column):
    """Crea un box plot."""

    fig = px.box(
        df,
        y=column,
        title=f"Box plot di {column}",
        template="plotly_white"
    )

    fig.update_layout(
        yaxis_title=column
    )

    return fig



# ============================================================
# 5. INTESTAZIONE
# ============================================================

st.title("📊 Data Analysis Dashboard")

st.caption(
    "Carica un dataset ed esplora le singole variabili "
    "attraverso grafici semplici."
)


# ============================================================
# 6. CARICAMENTO DEL DATASET
# ============================================================

st.sidebar.header("Caricamento dati")

uploaded_file = st.sidebar.file_uploader(
    "Seleziona un file CSV o Parquet",
    type=["csv", "parquet"]
)

df = None

if uploaded_file is not None:

    try:
        df = load_dataset(
            uploaded_file.getvalue(),
            uploaded_file.name
        )

    except Exception as error:
        st.error(f"Errore durante il caricamento: {error}")


# ============================================================
# 7. NAVIGAZIONE
# ============================================================

tab_eda, tab_report = st.tabs([
    "Analisi esplorativa",
    "Report analisi"
])


# ============================================================
# 8. TAB 1: ANALISI ESPLORATIVA
# ============================================================

with tab_eda:

    st.header("Analisi esplorativa")

    # Mostra le istruzioni se non è stato caricato un file.
    if df is None:

        st.info(
            "Carica un file CSV o Parquet dalla barra laterale "
            "per iniziare."
        )

    elif df.empty or len(df.columns) == 0:

        st.warning(
            "Il dataset è vuoto oppure non contiene colonne."
        )

    else:

        # ----------------------------------------------------
        # 8.1 Informazioni generali
        # ----------------------------------------------------

        st.subheader("Informazioni sul dataset")

        col1, col2, col3 = st.columns(3)

        col1.metric("Righe", f"{df.shape[0]:,}")
        col2.metric("Colonne", f"{df.shape[1]:,}")
        col3.metric(
            "Valori mancanti",
            f"{int(df.isna().sum().sum()):,}"
        )

        # ----------------------------------------------------
        # 8.2 Anteprima dei dati
        # ----------------------------------------------------

        st.subheader("Anteprima")

        st.dataframe(
            df.head(10),
            use_container_width=True
        )

        # ----------------------------------------------------
        # 8.3 Informazioni sulle colonne
        # ----------------------------------------------------

        with st.expander("Visualizza le colonne"):

            column_info = pd.DataFrame({
                "Colonna": df.columns,
                "Tipo": df.dtypes.astype(str).values,
                "Valori mancanti": df.isna().sum().values,
                "Valori distinti": df.nunique().values
            })

            st.dataframe(
                column_info,
                use_container_width=True,
                hide_index=True
            )

        # ----------------------------------------------------
        # 8.4 Selezione del grafico
        # ----------------------------------------------------

        st.subheader("Grafici")

        plot_type = st.selectbox(
            "Tipo di grafico",
            [
                "Istogramma",
                "Grafico a barre",
                "Box plot",
                "Istogramma categorizzato"
            ]
        )

        # Le colonne disponibili dipendono dal grafico.
        if plot_type in ["Istogramma", "Box plot"]:

            available_columns = get_numeric_columns(df)

        else:

            available_columns = df.columns.tolist()

        if not available_columns:

            st.warning(
                "Non sono presenti colonne compatibili "
                "con il grafico selezionato."
            )

        else:

            selected_column = st.selectbox(
                "Seleziona una colonna",
                available_columns
            )

            # Genera il grafico selezionato.
            if plot_type == "Istogramma":

                fig = plot_histogram(
                    df,
                    selected_column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            elif plot_type == "Grafico a barre":

                fig = plot_bar_chart(
                    df,
                    selected_column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            elif plot_type == "Box plot":

                fig = plot_boxplot(
                    df,
                    selected_column
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ----------------------------------------------------
        # 8.5 Valori mancanti
        # ----------------------------------------------------

        st.subheader("Valori mancanti")

        missing_values = (
            df.isna()
            .sum()
            .rename("Valori mancanti")
            .rename_axis("Colonna")
            .reset_index()
        )

        missing_values = missing_values[
            missing_values["Valori mancanti"] > 0
        ]

        if missing_values.empty:

            st.success(
                "Non sono presenti valori mancanti."
            )

        else:

            st.dataframe(
                missing_values,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# 9. TAB 2: REPORT ANALISI
# ============================================================

with tab_report:

    # Sezione riservata ai grafici generati da altri script.
    st.header("Report analisi")

    # In futuro, qui verranno visualizzati i grafici
    # prodotti dal codice di analisi precedente.

    pass


# ============================================================
# 10. FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "Data Analysis Dashboard"
)