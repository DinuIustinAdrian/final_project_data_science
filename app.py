import asyncio
import io
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import panel as pn
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

try:
    import joblib
    JOBLIB_DISPONIBILE = True
except ImportError:
    JOBLIB_DISPONIBILE = False

try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    import statsmodels.api as sm
    from scipy.stats import randint
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        precision_score, recall_score, f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.model_selection import (
        GridSearchCV, RandomizedSearchCV, RepeatedStratifiedKFold,
        StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        LabelEncoder,
        OneHotEncoder,
        StandardScaler,
    )
    from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

    SKLEARN_DISPONIBILE = True
except ImportError:
    SKLEARN_DISPONIBILE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_DISPONIBILE = True
except ImportError:
    CATBOOST_DISPONIBILE = False

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


# ============================================================
# CONFIGURAZIONE
# ============================================================

TITOLO = "Data Explorer & Model Dashboard"
NUMERO_GRAFICI = 6
MAX_CATEGORIE = 20
MAX_RIGHE_GRAFICO = 50000
MAX_RIGHE_TABELLA = 2000
MAX_PUNTI_OUTLIER = 5000
MAX_CLASSI = 20
MAX_CATEGORIE_FEATURE = 50

PAR_DIR = "data/"
PLOTS_DIR = "data/plots/"

# Modelli usati in main.py
MODELLI = [
    "Regressione logistica",
    "Albero decisionale",
    "Random forest",
    "CatBoost"
]

STILE_SCHEDA = {
    "border": "1px solid #dddddd",
    "border-radius": "8px",
    "padding": "10px",
    "background": "#ffffff"
}


# ============================================================
# STATO DELL'APPLICAZIONE
# ============================================================

stato = {
    "originale": None,
    "pulito": None,
    "nome_file": "",
    "registro": []
}


# ============================================================
# LETTURA CSV
# ============================================================

def leggi_csv(contenuto):
    """
    Legge un CSV provando diversi encoding e separatori.
    Restituisce un DataFrame.
    """
    if contenuto is None or len(contenuto) == 0:
        raise ValueError("Il file è vuoto.")

    separatori = [None, ",", ";", "	", "|"]
    encoding = ["utf-8-sig", "utf-8", "cp1252", "latin1"]

    ultimo_errore = None

    for codifica in encoding:
        for separatore in separatori:
            try:
                df = pd.read_csv(
                    io.BytesIO(contenuto),
                    sep=separatore,
                    engine="python",
                    encoding=codifica,
                    on_bad_lines="error"
                )

                if len(df.columns) == 0:
                    continue

                df.columns = [
                    str(colonna).strip()
                    for colonna in df.columns
                ]

                colonne_valide = [
                    colonna
                    for colonna in df.columns
                    if colonna and not colonna.startswith("Unnamed:")
                ]

                df = df.loc[:, colonne_valide]

                if len(df.columns) == 0:
                    continue

                if len(df.columns) == 1:
                    valore = str(df.columns[0])
                    if any(
                        simbolo in valore
                        for simbolo in [",", ";", "	", "|"]
                    ):
                        continue

                return df

            except Exception as errore:
                ultimo_errore = errore

    raise ValueError(
        f"Impossibile leggere il CSV. "
        f"Controlla il formato del file. Dettaglio: {ultimo_errore}"
    )


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def maschera_mancanti(df):
    """
    Restituisce un DataFrame di True/False con True dove il valore
    è mancante. Le stringhe vuote o fatte solo di spaces sono considerate
    mancanti.
    """
    maschera = df.isna()

    for colonna in df.columns:
        if (
            pd.api.types.is_object_dtype(df[colonna])
            or isinstance(df[colonna].dtype, pd.StringDtype)
        ):
            vuote = df[colonna].astype(str).str.strip() == ""
            maschera[colonna] = maschera[colonna] | vuote

    return maschera


def crea_tabella(df, totale=None, con_indice=True):
    """
    Crea una tabella (sola lettura) con paginazione locale.
    Mostra al massimo MAX_RIGHE_TABELLA righe.
    """
    if df is None or df.empty:
        return pn.pane.Markdown("_Nessuna riga da mostrare._")

    if totale is None:
        totale = len(df)

    visibile = df.head(MAX_RIGHE_TABELLA).copy()

    if con_indice:
        visibile.insert(0, "Indice riga", visibile.index)

    oggetti = []

    if totale > len(visibile):
        oggetti.append(
            pn.pane.Markdown(
                f"_Mostrate le prime {len(visibile)} righe su {totale}._"
            )
        )

    oggetti.append(
        pn.widgets.Tabulator(
            visibile,
            show_index=False,
            disabled=True,
            pagination="local",
            page_size=8,
            height=340,
            sizing_mode="stretch_width"
        )
    )

    return pn.Column(*oggetti, sizing_mode="stretch_width")


def mostra_esito(pannello, testo, tipo="info"):
    """Mostra un messaggio colorato (success, info, warning, danger)."""
    pannello.object = testo
    pannello.alert_type = tipo
    pannello.visible = True


def crea_area_conferma():
    """
    Crea un riquadro di conferma (Sì / Annulla).
    """
    messaggio_conferma = pn.pane.Alert(
        "",
        alert_type="warning",
        sizing_mode="stretch_width"
    )

    pulsante_si = pn.widgets.Button(
        name="Sì, conferma",
        button_type="danger"
    )

    pulsante_no = pn.widgets.Button(
        name="Annulla",
        button_type="light"
    )

    contenitore = pn.Column(
        messaggio_conferma,
        pn.Row(pulsante_si, pulsante_no),
        visible=False,
        sizing_mode="stretch_width"
    )

    azione = {"funzione": None}

    def chiedi(testo, funzione):
        messaggio_conferma.object = testo
        azione["funzione"] = funzione
        contenitore.visible = True

    def nascondi(event=None):
        azione["funzione"] = None
        contenitore.visible = False

    def conferma(event):
        funzione = azione["funzione"]
        nascondi()
        if funzione is not None:
            funzione()

    pulsante_si.on_click(conferma)
    pulsante_no.on_click(nascondi)

    return contenitore, chiedi, nascondi


# ============================================================
# FUNZIONI PER I PLOT RECAP DI MAIN.PY (HOMEPAGE)
# ============================================================

def carica_pane_plot(base_name):
    """
    Tenta di caricare il grafico in formato SVG se disponibile, altrimenti PNG.
    """
    svg_file = os.path.join(PLOTS_DIR, f"{base_name}.svg")
    png_file = os.path.join(PLOTS_DIR, f"{base_name}.png")

    if os.path.exists(svg_file):
        try:
            with open(svg_file, "r", encoding="utf-8") as f:
                content = f.read()
            return pn.pane.SVG(content, sizing_mode="stretch_width", height=380), "SVG"
        except Exception:
            pass

    if os.path.exists(png_file):
        try:
            return pn.pane.PNG(png_file, sizing_mode="stretch_width", height=380), "PNG"
        except Exception:
            pass

    return None, None


PLOT_DEFINIZIONI = [
    ("Confronto Curve ROC (Modelli)", "plot_confronto_roc"),
    ("Confronto Metriche Modelli", "plot_confronto_metriche"),
    ("Stato di Depressione Complessivo", "plot_DepressionStatus"),
    ("Distribuzione Età per Genere", "plot_AgeDistribution_Gender"),
    ("Matrice V di Cramér", "plot_vCramer"),
    ("Correlazione di Spearman", "plot_correlazione_spearman"),
    ("Costo Extra-Studio vs Depressione (Quintili)", "plot_extra_studio_vs_depressione"),
    ("CGPA vs Depressione (Quintili)", "plot_cgpa_quintili_depressione"),
    ("Stato Relazione x Cerchia Amici", "plot_Relationship_FriendCircle"),
    ("Albero di Decisione (Primi 3 livelli)", "plot_decision_tree"),
    ("Feature Importance - Decision Tree", "plot_feature_importance"),
    ("Feature Importance - Random Forest", "plot_rf_feature_importance"),
    ("Feature Importance - CatBoost", "plot_catboost_feature_importance"),
    ("Ottimizzazione Soglia Random Forest", "plot_rf_threshold_tuning"),
    ("Curva Alberi CatBoost (Train vs CV)", "plot_catboost_curva_alberi"),
    ("Andamento OOB Random Forest", "plot_rf_oob_curve"),
    ("Family Background vs Depression", "plot_Family_background_vs_Depression"),
    ("Living Style vs Depression", "plot_Living_style_vs_Depression"),
    ("Relationship Status vs Depression", "plot_Relationship_status_vs_Depression"),
]

report_selettore = pn.widgets.Select(
    name="Grafico da visualizzare", options={}, sizing_mode="stretch_width"
)
report_visualizzazione = pn.Column(sizing_mode="stretch_width")


def mostra_report(event=None):
    """Mostra il grafico associato al valore selezionato nel widget."""
    disponibili = report_selettore.options or {}
    base_name = report_selettore.value

    if not disponibili:
        report_visualizzazione.objects = [pn.pane.Alert(
            "Nessun grafico disponibile. Esegui main.py per generare i report.",
            alert_type="info", sizing_mode="stretch_width"
        )]
        return

    # In un Select con options={etichetta: valore}, value contiene il valore,
    # quindi il nome del file e non l'etichetta mostrata nel menu.
    titolo = next(
        (etichetta for etichetta, valore in disponibili.items()
         if valore == base_name),
        None
    )

    if titolo is None:
        return

    pane_img, formato = carica_pane_plot(base_name)

    if pane_img is None:
        report_visualizzazione.objects = [pn.pane.Alert(
            f"Impossibile caricare il grafico '{titolo}'.",
            alert_type="danger", sizing_mode="stretch_width"
        )]
    else:
        report_visualizzazione.objects = [pn.Column(
            pn.pane.Markdown(f"### {titolo}  ·  `{formato}`"),
            pane_img,
            styles=STILE_SCHEDA,
            sizing_mode="stretch_width"
        )]


def aggiorna_report(event=None):
    """Aggiorna l'elenco dei report preservando il grafico selezionato."""
    selezione_precedente = report_selettore.value

    if not os.path.exists(PLOTS_DIR):
        report_selettore.options = {}
        report_visualizzazione.objects = [pn.pane.Alert(
            "La cartella `data/plots/` non è ancora disponibile. Esegui main.py per generare i report.",
            alert_type="info", sizing_mode="stretch_width"
        )]
        return

    disponibili = {}
    for titolo, base_name in PLOT_DEFINIZIONI:
        if (
            os.path.exists(os.path.join(PLOTS_DIR, f"{base_name}.svg"))
            or os.path.exists(os.path.join(PLOTS_DIR, f"{base_name}.png"))
        ):
            disponibili[titolo] = base_name

    # Aggiorna le opzioni solo quando sono effettivamente cambiate.
    if report_selettore.options != disponibili:
        report_selettore.options = disponibili

    if not disponibili:
        report_visualizzazione.objects = [pn.pane.Alert(
            "Nessun grafico trovato in `data/plots/`.",
            alert_type="warning", sizing_mode="stretch_width"
        )]
        return

    valori_disponibili = list(disponibili.values())
    if selezione_precedente in valori_disponibili:
        valore_da_selezionare = selezione_precedente
    else:
        valore_da_selezionare = valori_disponibili[0]

    if report_selettore.value != valore_da_selezionare:
        report_selettore.value = valore_da_selezionare
    else:
        # Se il valore non cambia, il watcher non scatta: aggiorniamo il plot.
        mostra_report()


# Il watcher deve aggiornare il contenuto, non ricostruire l'elenco delle opzioni.
report_selettore.param.watch(mostra_report, "value")


def crea_recap_plot_main():
    """Costruisce il tab Report con selezione del singolo grafico."""
    aggiorna_report()
    return pn.Column(
        pn.pane.Markdown("## Report grafici\nSeleziona dal menu il report che vuoi consultare. Viene mostrato un grafico alla volta."),
        report_selettore,
        report_visualizzazione,
        sizing_mode="stretch_width"
    )


# ============================================================
# STATISTICHE DEL DATASET
# ============================================================

def crea_statistiche(df):
    numero_righe = len(df)
    numero_colonne = len(df.columns)

    maschera = maschera_mancanti(df)
    celle_mancanti = int(maschera.sum().sum())
    totale_celle = numero_righe * numero_colonne

    percentuale_mancanti = (celle_mancanti / totale_celle * 100) if totale_celle > 0 else 0
    righe_incomplete = int(maschera.any(axis=1).sum())
    duplicati = int(df.duplicated().sum())

    colonne_numeriche = len(df.select_dtypes(include="number").columns)
    colonne_categoriche = numero_colonne - colonne_numeriche

    statistiche = [
        ("Righe", numero_righe),
        ("Colonne", numero_colonne),
        ("Valori mancanti", celle_mancanti),
        ("Mancanti %", f"{percentuale_mancanti:.2f}%"),
        ("Righe incomplete", righe_incomplete),
        ("Duplicati", duplicati),
        ("Colonne numeriche", colonne_numeriche),
        ("Colonne non numeriche", colonne_categoriche),
    ]

    schede = []
    for titolo, valore in statistiche:
        scheda = pn.pane.HTML(
            f"""
            <div style="
                border: 1px solid #dddddd;
                border-radius: 8px;
                padding: 15px;
                text-align: center;
                background-color: #ffffff;
                min-height: 85px;
            ">
                <div style="font-size: 13px; color: #666666; margin-bottom: 8px;">
                    {titolo}
                </div>
                <div style="font-size: 24px; font-weight: bold; color: #222222;">
                    {valore}
                </div>
            </div>
            """,
            sizing_mode="stretch_width"
        )
        schede.append(scheda)

    return pn.GridBox(*schede, ncols=4, sizing_mode="stretch_width")


# ============================================================
# STATISTICHE DESCRITTIVE
# ============================================================

def crea_tabella_statistiche(df):
    if df.empty or len(df.columns) == 0:
        return pn.pane.Markdown("Nessun dato disponibile.")

    statistiche = df.describe(include="all").transpose().reset_index()
    statistiche = statistiche.rename(columns={"index": "Variabile"})

    return pn.widgets.Tabulator(
        statistiche,
        show_index=False,
        pagination="local",
        page_size=15,
        height=350,
        sizing_mode="stretch_width"
    )


# ============================================================
# CREAZIONE DEL GRAFICO
# ============================================================

def crea_grafico(df, colonna, gruppo=None):
    if colonna not in df.columns:
        return px.scatter(title="Seleziona una variabile valida")

    dati = df.copy()
    nota_campione = ""

    if len(dati) > MAX_RIGHE_GRAFICO:
        dati = dati.sample(n=MAX_RIGHE_GRAFICO, random_state=42)
        nota_campione = f" (campione di {MAX_RIGHE_GRAFICO} righe)"

    for nome_colonna in dati.columns:
        if pd.api.types.is_object_dtype(dati[nome_colonna]) or isinstance(dati[nome_colonna].dtype, pd.StringDtype):
            dati[nome_colonna] = dati[nome_colonna].replace(r"^\s*$", np.nan, regex=True)

    dati = dati.dropna(subset=[colonna])
    if dati.empty:
        return px.scatter(title=f"Nessun dato disponibile per {colonna}")

    variabile_numerica = pd.api.types.is_numeric_dtype(dati[colonna])

    if not variabile_numerica:
        conversione = pd.to_numeric(dati[colonna], errors="coerce")
        valori_originali = dati[colonna].notna().sum()
        valori_convertiti = conversione.notna().sum()
        if valori_originali > 0 and valori_convertiti / valori_originali >= 0.9:
            dati[colonna] = conversione
            dati = dati.dropna(subset=[colonna])
            variabile_numerica = True

    if gruppo == colonna or gruppo not in dati.columns:
        gruppo = None

    if variabile_numerica:
        if gruppo is not None:
            dati = dati.dropna(subset=[gruppo])
            if dati.empty:
                return px.scatter(title="Nessun dato disponibile per il confronto")

            dati[gruppo] = dati[gruppo].astype(str)
            gruppi_principali = dati[gruppo].value_counts().head(MAX_CATEGORIE).index
            dati = dati[dati[gruppo].isin(gruppi_principali)].copy()

            fig = px.histogram(
                dati,
                x=colonna,
                color=gruppo,
                barmode="group",
                opacity=0.65,
                nbins=30,
                title=f"Distribuzione di {colonna} per {gruppo}{nota_campione}",
                labels={colonna: colonna, gruppo: gruppo}
            )
            fig.update_layout(yaxis_title="Frequenza", legend_title_text=gruppo)
        else:
            fig = px.histogram(
                dati,
                x=colonna,
                nbins=30,
                title=f"Distribuzione di {colonna}{nota_campione}"
            )
            fig.update_layout(yaxis_title="Frequenza")
    else:
        dati[colonna] = dati[colonna].astype(str)
        categorie_principali = dati[colonna].value_counts().head(MAX_CATEGORIE).index
        dati = dati[dati[colonna].isin(categorie_principali)].copy()

        if gruppo is not None:
            dati[gruppo] = dati[gruppo].fillna("(Mancante)").astype(str)
            gruppi_principali = dati[gruppo].value_counts().head(MAX_CATEGORIE).index
            dati = dati[dati[gruppo].isin(gruppi_principali)].copy()

            conteggi = pd.crosstab(dati[colonna], dati[gruppo])
            fig = px.bar(
                conteggi,
                barmode="group",
                title=f"{colonna} per {gruppo}{nota_campione}",
                labels={"index": colonna, colonna: colonna, "value": "Frequenza"}
            )
            fig.update_layout(xaxis_title=colonna, yaxis_title="Frequenza", legend_title_text=gruppo)
        else:
            conteggi = dati[colonna].value_counts().reset_index()
            conteggi.columns = [colonna, "Frequenza"]
            fig = px.bar(
                conteggi,
                x=colonna,
                y="Frequenza",
                title=f"Frequenza di {colonna}{nota_campione}"
            )
            fig.update_layout(xaxis_title=colonna, yaxis_title="Frequenza")

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=60),
        font=dict(size=11)
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


# ============================================================
# TIPI DI GRAFICO A DUE VARIABILI
# ============================================================

TIPO_AUTOMATICO = "Automatico (istogramma / barre)"
TIPO_SCATTER = "Scatter plot"
TIPO_LINEE = "Linee"
TIPO_BOX = "Box plot"
TIPO_BARRE = "Barre (aggregazione)"
TIPO_HEATMAP = "Heatmap di densità"

TIPI_GRAFICO = {
    TIPO_AUTOMATICO: {"y": False, "gruppo": True, "aggregazione": False},
    TIPO_SCATTER: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_LINEE: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_BOX: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_BARRE: {"y": True, "gruppo": True, "aggregazione": True},
    TIPO_HEATMAP: {"y": True, "gruppo": False, "aggregazione": False},
}

FUNZIONI_AGGREGAZIONE = {"Media": "mean", "Mediana": "median", "Somma": "sum"}
PALETTE_COLORI = px.colors.qualitative.Plotly


def figura_vuota(testo):
    return px.scatter(title=testo)


def converti_in_numerica(dati, colonna):
    if pd.api.types.is_numeric_dtype(dati[colonna]):
        return True
    conversione = pd.to_numeric(dati[colonna], errors="coerce")
    valori_originali = dati[colonna].notna().sum()
    valori_convertiti = conversione.notna().sum()
    if valori_originali > 0 and valori_convertiti / valori_originali >= 0.9:
        dati[colonna] = conversione
        return True
    return False


def limita_categorie(dati, colonna):
    dati = dati.copy()
    dati[colonna] = dati[colonna].astype(str)
    principali = dati[colonna].value_counts().head(MAX_CATEGORIE).index
    return dati[dati[colonna].isin(principali)].copy()


def impostazioni_comuni(fig):
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=60),
        font=dict(size=11)
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def crea_grafico_due_variabili(df, tipo, colonna_x, colonna_y, gruppo=None, aggregazione="Media"):
    if colonna_x not in df.columns or colonna_y not in df.columns or colonna_x == colonna_y:
        return figura_vuota("Seleziona due variabili distinte")

    if tipo == TIPO_HEATMAP or gruppo in (colonna_x, colonna_y) or gruppo not in df.columns:
        gruppo = None

    colonne_usate = [colonna_x, colonna_y]
    if gruppo is not None:
        colonne_usate.append(gruppo)

    dati = df[colonne_usate].copy()
    nota_campione = ""

    if len(dati) > MAX_RIGHE_GRAFICO:
        dati = dati.sample(n=MAX_RIGHE_GRAFICO, random_state=42)
        nota_campione = f" (campione di {MAX_RIGHE_GRAFICO} righe)"

    for nome_colonna in dati.columns:
        if pd.api.types.is_object_dtype(dati[nome_colonna]) or isinstance(dati[nome_colonna].dtype, pd.StringDtype):
            dati[nome_colonna] = dati[nome_colonna].replace(r"^\s*$", np.nan, regex=True)

    if not converti_in_numerica(dati, colonna_y):
        return figura_vuota(f"{tipo}: la variabile Y «{colonna_y}» deve essere numerica")

    if tipo in (TIPO_SCATTER, TIPO_HEATMAP):
        if not converti_in_numerica(dati, colonna_x):
            return figura_vuota(f"{tipo}: la variabile X «{colonna_x}» deve essere numerica")

    if tipo == TIPO_LINEE:
        if not converti_in_numerica(dati, colonna_x):
            date = pd.to_datetime(dati[colonna_x], errors="coerce")
            valide = dati[colonna_x].notna().sum()
            if valide > 0 and date.notna().sum() >= 0.9 * valide:
                dati[colonna_x] = date
            else:
                return figura_vuota(f"{tipo}: la variabile X «{colonna_x}» deve essere numerica o una data")

    dati = dati.dropna(subset=colonne_usate)
    if dati.empty:
        return figura_vuota("Nessun dato disponibile per questa combinazione")

    if gruppo is not None:
        dati = limita_categorie(dati, gruppo)

    per_gruppo = f" per {gruppo}" if gruppo is not None else ""

    if tipo == TIPO_SCATTER:
        fig = px.scatter(
            dati, x=colonna_x, y=colonna_y, color=gruppo, opacity=0.7,
            title=f"{colonna_y} in funzione di {colonna_x}{per_gruppo}{nota_campione}"
        )
    elif tipo == TIPO_LINEE:
        dati = dati.sort_values(colonna_x)
        fig = px.line(
            dati, x=colonna_x, y=colonna_y, color=gruppo,
            title=f"{colonna_y} in funzione di {colonna_x}{per_gruppo}{nota_campione}"
        )
    elif tipo == TIPO_BOX:
        dati = limita_categorie(dati, colonna_x)
        fig = px.box(
            dati, x=colonna_x, y=colonna_y, color=gruppo,
            title=f"Distribuzione di {colonna_y} per {colonna_x}{per_gruppo}{nota_campione}"
        )
    elif tipo == TIPO_BARRE:
        dati = limita_categorie(dati, colonna_x)
        funzione = FUNZIONI_AGGREGAZIONE.get(aggregazione, "mean")
        colonne_raggruppamento = [colonna_x]
        if gruppo is not None:
            colonne_raggruppamento.append(gruppo)

        sintesi = dati.groupby(colonne_raggruppamento)[colonna_y].agg(funzione).reset_index()
        etichetta = f"{aggregazione} di {colonna_y}"
        sintesi = sintesi.rename(columns={colonna_y: etichetta})

        fig = px.bar(
            sintesi, x=colonna_x, y=etichetta, color=gruppo, barmode="group",
            title=f"{etichetta} per {colonna_x}{per_gruppo}{nota_campione}"
        )
    else:
        fig = px.density_heatmap(
            dati, x=colonna_x, y=colonna_y, nbinsx=30, nbinsy=30,
            color_continuous_scale=["#ffffff", "#2563eb"],
            title=f"Densità di {colonna_y} e {colonna_x}{nota_campione}"
        )
        fig.update_layout(coloraxis_colorbar_title_text="Frequenza")

    return impostazioni_comuni(fig)


def crea_grafico_scelto(df, tipo, colonna_x, colonna_y, gruppo, aggregazione):
    try:
        if tipo == TIPO_AUTOMATICO:
            return crea_grafico(df, colonna_x, gruppo)
        return crea_grafico_due_variabili(df, tipo, colonna_x, colonna_y, gruppo, aggregazione)
    except Exception as errore:
        return figura_vuota(f"Impossibile creare il grafico: {errore}")


# ============================================================
# COLORI DEI GRAFICI
# ============================================================

def colore_predefinito(traccia, posizione):
    if traccia.type == "histogram2d":
        return "#2563eb"

    candidati = []
    try:
        candidati.append(traccia.marker.color)
    except Exception:
        pass
    try:
        candidati.append(traccia.line.color)
    except Exception:
        pass

    for candidato in candidati:
        if isinstance(candidato, str) and candidato.startswith("#") and len(candidato) == 7:
            return candidato

    return PALETTE_COLORI[posizione % len(PALETTE_COLORI)]


def applica_colori(fig, colori):
    for traccia in fig.data:
        etichetta = traccia.name or "Colore"
        colore = colori.get(etichetta)
        if colore is None:
            continue

        if traccia.type == "histogram2d":
            fig.update_layout(coloraxis_colorscale=[[0, "#ffffff"], [1, colore]])
        elif traccia.type in ("scatter", "scattergl"):
            traccia.marker.color = colore
            if traccia.mode is not None and "lines" in traccia.mode:
                traccia.line.color = colore
        elif traccia.type == "box":
            traccia.marker.color = colore
            traccia.line.color = colore
        else:
            traccia.marker.color = colore


# ============================================================
# SINGOLA SCHEDA GRAFICO
# ============================================================

def crea_scheda_grafico(df, colonne, numero):
    indice = (numero - 1) % len(colonne)
    colonna_iniziale = colonne[indice]

    colori_scelti = {}
    blocco = {"attivo": False}

    selettore_tipo = pn.widgets.Select(
        name="Tipo di grafico", options=list(TIPI_GRAFICO.keys()), value=TIPO_AUTOMATICO, sizing_mode="stretch_width"
    )
    selettore_variabile = pn.widgets.Select(
        name="Variabile", options=colonne, value=colonna_iniziale, sizing_mode="stretch_width"
    )

    opzioni_y = [c for c in colonne if c != colonna_iniziale]
    colonna_y_iniziale = colonne[(indice + 1) % len(colonne)]
    if colonna_y_iniziale not in opzioni_y:
        colonna_y_iniziale = opzioni_y[0] if opzioni_y else None

    selettore_y = pn.widgets.Select(
        name="Variabile Y", options=opzioni_y, value=colonna_y_iniziale, visible=False, sizing_mode="stretch_width"
    )

    opzioni_gruppo = ["Nessuno"] + [c for c in colonne if c != colonna_iniziale]
    selettore_gruppo = pn.widgets.Select(
        name="Confronta per", options=opzioni_gruppo, value="Nessuno", sizing_mode="stretch_width"
    )

    selettore_aggregazione = pn.widgets.Select(
        name="Aggregazione", options=list(FUNZIONI_AGGREGAZIONE.keys()), value="Media", visible=False, sizing_mode="stretch_width"
    )

    grafico = pn.pane.Plotly(
        figura_vuota(""), height=320, sizing_mode="stretch_width",
        config={"responsive": True, "displayModeBar": True, "scrollZoom": True}
    )

    pulsante_ripristina_colori = pn.widgets.Button(name="Ripristina colori predefiniti", button_type="light")
    scheda_colori = pn.Card(title="Colori", collapsed=True, sizing_mode="stretch_width")

    def cambia_colore(event):
        colori_scelti[event.obj.name] = event.new
        applica_colori(grafico.object, colori_scelti)
        grafico.param.trigger("object")

    def ripristina_colori(event=None):
        colori_scelti.clear()
        ridisegna()

    pulsante_ripristina_colori.on_click(ripristina_colori)

    def ridisegna(event=None):
        if blocco["attivo"]:
            return

        tipo = selettore_tipo.value
        impostazioni = TIPI_GRAFICO[tipo]
        colonna = selettore_variabile.value
        colonna_y = selettore_y.value if impostazioni["y"] else None
        gruppo = selettore_gruppo.value if impostazioni["gruppo"] and selettore_gruppo.value != "Nessuno" else None

        fig = crea_grafico_scelto(df, tipo, colonna, colonna_y, gruppo, selettore_aggregazione.value)

        predefiniti = {}
        for posizione, traccia in enumerate(fig.data):
            etichetta = traccia.name or "Colore"
            if etichetta not in predefiniti:
                predefiniti[etichetta] = colore_predefinito(traccia, posizione)

        applica_colori(fig, colori_scelti)
        grafico.object = fig

        selettori_colore = []
        for etichetta, predefinito in predefiniti.items():
            selettore = pn.widgets.ColorPicker(name=etichetta, value=colori_scelti.get(etichetta, predefinito), width=130)
            selettore.param.watch(cambia_colore, "value")
            selettori_colore.append(selettore)

        if len(selettori_colore) > 0:
            scheda_colori.objects = [pn.GridBox(*selettori_colore, ncols=2), pulsante_ripristina_colori]
        else:
            scheda_colori.objects = [pn.pane.Markdown("_Nessun elemento da colorare._")]

    def cambia_variabili(event=None):
        if blocco["attivo"]:
            return
        blocco["attivo"] = True
        try:
            tipo = selettore_tipo.value
            impostazioni = TIPI_GRAFICO[tipo]
            usa_y = impostazioni["y"]

            selettore_variabile.name = "Variabile X" if usa_y else "Variabile"
            selettore_y.visible = usa_y
            selettore_gruppo.visible = impostazioni["gruppo"]
            selettore_aggregazione.visible = impostazioni["aggregazione"]

            colonna = selettore_variabile.value
            nuove_opzioni_y = [nome for nome in colonne if nome != colonna]
            selettore_y.options = nuove_opzioni_y

            if selettore_y.value not in nuove_opzioni_y:
                selettore_y.value = nuove_opzioni_y[0] if len(nuove_opzioni_y) > 0 else None

            escluse = [colonna]
            if usa_y:
                escluse.append(selettore_y.value)

            nuove_opzioni_gruppo = ["Nessuno"] + [nome for nome in colonne if nome not in escluse]
            valore_precedente = selettore_gruppo.value
            selettore_gruppo.options = nuove_opzioni_gruppo

            if valore_precedente not in nuove_opzioni_gruppo:
                selettore_gruppo.value = "Nessuno"
        finally:
            blocco["attivo"] = False

        ridisegna()

    selettore_tipo.param.watch(cambia_variabili, "value")
    selettore_variabile.param.watch(cambia_variabili, "value")
    selettore_y.param.watch(cambia_variabili, "value")
    selettore_gruppo.param.watch(ridisegna, "value")
    selettore_aggregazione.param.watch(ridisegna, "value")

    ridisegna()

    return pn.Column(
        pn.pane.Markdown(f"### Grafico {numero}"),
        selettore_tipo,
        pn.Row(selettore_variabile, selettore_y, sizing_mode="stretch_width"),
        pn.Row(selettore_gruppo, selettore_aggregazione, sizing_mode="stretch_width"),
        grafico,
        scheda_colori,
        styles=STILE_SCHEDA,
        sizing_mode="stretch_width"
    )


# ============================================================
# CREAZIONE DI TUTTI I GRAFICI
# ============================================================

def crea_istogrammi(df):
    colonne = list(df.columns)
    if len(colonne) == 0:
        return pn.pane.Markdown("Il dataset non contiene colonne da visualizzare.")

    schedes = [crea_scheda_grafico(df, colonne, numero) for numero in range(1, NUMERO_GRAFICI + 1)]
    return pn.GridBox(*schedes, ncols=3, sizing_mode="stretch_width")


# ============================================================
# COMPONENTI PRINCIPALI DELLA DASHBOARD
# ============================================================

file_input = pn.widgets.FileInput(name="Carica CSV", accept=".csv", sizing_mode="stretch_width")

messaggio = pn.pane.Markdown(
    "## Benvenuto in Data Explorer\n\n"
    "Carica un file CSV dalla barra laterale per visualizzare la dashboard, esplorare i dati, "
    "applicare operazioni di pulizia e confrontare i modelli di classificazione."
)

nome_dataset = pn.pane.Markdown("")

esplorazione = pn.Column(sizing_mode="stretch_width")
dashboard_home = pn.Column(sizing_mode="stretch_width")
report_main = pn.Column(crea_recap_plot_main(), sizing_mode="stretch_width")
main = pn.Column(messaggio, sizing_mode="stretch_width")


# ============================================================
# SCHEDA "PULIZIA DATI": COMPONENTI
# ============================================================

testo_stato_pulizia = pn.pane.Markdown("")

barra_stato = pn.Column(
    testo_stato_pulizia,
    styles={
        "border": "1px solid #bfdbfe",
        "border-radius": "8px",
        "padding": "4px 12px",
        "background": "#eff6ff"
    },
    sizing_mode="stretch_width"
)

# Duplicati
dup_colonne = pn.widgets.MultiSelect(name="Colonne da confrontare", options=[], size=8, sizing_mode="stretch_width")
dup_mantieni = pn.widgets.Select(name="Da ogni gruppo di duplicati mantieni", options=["Prima occorrenza", "Ultima occorrenza"], value="Prima occorrenza", sizing_mode="stretch_width")
dup_cerca = pn.widgets.Button(name="Cerca duplicati", button_type="primary")
dup_rimuovi = pn.widgets.Button(name="Rimuovi duplicati…", button_type="danger", visible=False)
dup_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
dup_risultato = pn.Column(sizing_mode="stretch_width")
dup_conferma, dup_chiedi, dup_nascondi = crea_area_conferma()

# Valori mancanti
mis_statistiche = pn.Column(sizing_mode="stretch_width")
mis_colonne = pn.widgets.MultiSelect(name="Colonne da considerare", options=[], size=8, sizing_mode="stretch_width")
mis_mostra = pn.widgets.Button(name="Mostra righe interessate", button_type="primary")
mis_rimuovi = pn.widgets.Button(name="Rimuovi righe…", button_type="danger")
mis_metodo = pn.widgets.Select(name="Imputa i valori mancanti con", options=["Media", "Mediana", "Moda", "Valore personalizzato"], value="Media", sizing_mode="stretch_width")
mis_valore = pn.widgets.TextInput(name="Valore personalizzato", placeholder="Es. 0, N/D, Sconosciuto", visible=False, sizing_mode="stretch_width")
mis_imputa = pn.widgets.Button(name="Imputa valori", button_type="success")
mis_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
mis_risultato = pn.Column(sizing_mode="stretch_width")
mis_conferma, mis_chiedi, mis_nascondi = crea_area_conferma()

# Outlier
out_colonna = pn.widgets.Select(name="Variabile numerica", options=[], sizing_mode="stretch_width")
out_metodo = pn.widgets.Select(name="Metodo", options=["IQR (1,5 × IQR)", "Percentili 5° e 95°"], value="IQR (1,5 × IQR)", sizing_mode="stretch_width")
out_cerca = pn.widgets.Button(name="Individua outlier", button_type="primary")
out_rimuovi = pn.widgets.Button(name="Rimuovi outlier…", button_type="danger", visible=False)
out_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
out_risultato = pn.Column(sizing_mode="stretch_width")
out_conferma, out_chiedi, out_nascondi = crea_area_conferma()

# Colonne
col_informazioni = pn.Column(sizing_mode="stretch_width")
col_selezione = pn.widgets.MultiSelect(name="Colonne da rimuovere", options=[], size=10, sizing_mode="stretch_width")
col_rimuovi = pn.widgets.Button(name="Rimuovi colonne selezionate…", button_type="danger")
col_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
col_conferma, col_chiedi, col_nascondi = crea_area_conferma()

# Dataset pulito
riepilogo_dataset = pn.pane.Markdown("")
registro_operazioni = pn.pane.Markdown("")
ripristina = pn.widgets.Button(name="Ripristina dataset originale…", button_type="warning")
ripristino_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
ripristino_conferma, ripristino_chiedi, ripristino_nascondi = crea_area_conferma()

selettore_separatore = pn.widgets.Select(
    name="Separatore del CSV scaricato",
    options={"Virgola ( , )": ",", "Punto e virgola ( ; )": ";"},
    value=",",
    width=250
)


def genera_csv():
    df = stato["pulito"]
    if df is None:
        return None
    testo = df.to_csv(index=False, sep=selettore_separatore.value)
    return io.BytesIO(testo.encode("utf-8-sig"))


scarica = pn.widgets.FileDownload(
    callback=genera_csv,
    filename="dataset_pulito.csv",
    label="Scarica il dataset pulito (CSV)",
    button_type="success",
    width=300
)


# ============================================================
# SCHEDA "PULIZIA DATI": AGGIORNAMENTO DELL'INTERFACCIA
# ============================================================

def aggiorna_pulizia():
    df = stato["pulito"]
    originale = stato["originale"]

    if df is None or originale is None:
        return

    colonne = list(df.columns)
    maschera = maschera_mancanti(df)
    colonne_con_mancanti = [c for c in colonne if maschera[c].any()]
    colonne_numeriche = list(df.select_dtypes(include="number").columns)

    righe_rimosse = len(originale) - len(df)
    testo_stato_pulizia.object = (
        f"**Dataset originale:** {originale.shape[0]} righe × {originale.shape[1]} colonne &nbsp;|&nbsp; "
        f"**Dataset pulito:** {df.shape[0]} righe × {df.shape[1]} colonne &nbsp;|&nbsp; "
        f"**Righe rimosse:** {righe_rimosse}"
    )

    dup_colonne.value = []
    dup_colonne.options = colonne
    dup_colonne.value = colonne
    dup_nascondi()
    dup_esito.visible = False
    dup_rimuovi.visible = False
    dup_risultato.objects = []

    mis_colonne.value = []
    mis_colonne.options = colonne
    mis_colonne.value = colonne_con_mancanti
    mis_nascondi()
    mis_esito.visible = False
    mis_risultato.objects = []

    righe = len(df)
    conteggi = maschera.sum()
    tabella_mancanti = pd.DataFrame({
        "Colonna": colonne,
        "Tipo": [str(tipo) for tipo in df.dtypes],
        "Mancanti": [int(conteggi[col]) for col in colonne]
    })
    tabella_mancanti["Mancanti %"] = (tabella_mancanti["Mancanti"] / righe * 100).round(2) if righe > 0 else 0.0
    tabella_mancanti = tabella_mancanti.sort_values("Mancanti", ascending=False)

    totale_mancanti = int(maschera.sum().sum())
    righe_incomplete = int(maschera.any(axis=1).sum())

    mis_statistiche.objects = [
        pn.pane.Markdown(f"**{totale_mancanti}** valori mancanti in **{righe_incomplete}** righe incomplete (su {righe} totali)."),
        crea_tabella(tabella_mancanti, con_indice=False)
    ]

    out_colonna.options = colonne_numeriche
    out_colonna.value = colonne_numeriche[0] if len(colonne_numeriche) > 0 else None
    out_cerca.disabled = len(colonne_numeriche) == 0
    out_nascondi()
    out_esito.visible = False
    out_rimuovi.visible = False
    out_risultato.objects = []

    tabella_colonne = pd.DataFrame({
        "Colonna": colonne,
        "Tipo": [str(tipo) for tipo in df.dtypes],
        "Valori distinti": [int(df[col].nunique()) for col in colonne],
        "Mancanti": [int(conteggi[col]) for col in colonne],
        "Mancanti %": list(tabella_mancanti.set_index("Colonna").loc[colonne, "Mancanti %"])
    })

    col_informazioni.objects = [crea_tabella(tabella_colonne, con_indice=False)]
    col_selezione.value = []
    col_selezione.options = colonne
    col_nascondi()
    col_esito.visible = False

    riepilogo_dataset.object = (
        "| | Originale | Dataset pulito |\n"
        "|---|---:|---:|\n"
        f"| Righe | {originale.shape[0]} | {df.shape[0]} |\n"
        f"| Colonne | {originale.shape[1]} | {df.shape[1]} |\n"
        f"| Valori mancanti | {int(maschera_mancanti(originale).sum().sum())} | {totale_mancanti} |\n"
        f"| Righe duplicate | {int(originale.duplicated().sum())} | {int(df.duplicated().sum())} |\n"
    )

    if len(stato["registro"]) == 0:
        registro_operazioni.object = "_Nessuna operazione eseguita: il dataset pulito coincide con l'originale._"
    else:
        elenco = [f"{n}. {v}" for n, v in enumerate(stato["registro"], start=1)]
        registro_operazioni.object = "\n".join(elenco)

    ripristino_nascondi()
    ripristino_esito.visible = False


def aggiorna_esplorazione():
    """Ricostruisce la scheda Esplorazione con statistiche e grafici del CSV."""
    df = stato["pulito"]
    originale = stato["originale"]

    if df is None or originale is None:
        esplorazione.objects = [pn.pane.Markdown("Carica un CSV per iniziare l'esplorazione.")]
        return

    nuove_statistiche = crea_statistiche(df)
    nuova_tabella = crea_tabella_statistiche(df)
    nuovi_grafici = crea_istogrammi(df)

    nome_file = stato["nome_file"]

    nome_dataset.object = (
        f"### Dataset caricato: `{nome_file}`\n\n"
        f"Dimensioni originali: **{originale.shape[0]} righe** × **{originale.shape[1]} colonne** &nbsp;|&nbsp; "
        f"Dataset di lavoro: **{df.shape[0]} righe** × **{df.shape[1]} colonne**"
    )

    esplorazione.objects = [
        pn.pane.Markdown("## Panoramica del dataset"),
        nuove_statistiche,
        pn.layout.Divider(),
        pn.pane.Markdown("## Statistiche descrittive"),
        nuova_tabella,
        pn.layout.Divider(),
        pn.pane.Markdown("## Esplorazione interattiva delle variabili"),
        nuovi_grafici
    ]


def aggiorna_dashboard_home():
    'Aggiorna la dashboard compatta del dataset Student Depression.'
    df = stato["pulito"]
    if df is None:
        dashboard_home.objects = [messaggio]
        return

    maschera = maschera_mancanti(df)
    celle_mancanti = int(maschera.sum().sum())
    totale_celle = int(df.shape[0] * df.shape[1])
    percentuale_mancanti = celle_mancanti / totale_celle * 100 if totale_celle else 0
    duplicati = int(df.duplicated().sum())
    numeriche = len(df.select_dtypes(include="number").columns)
    categoriche = len(df.columns) - numeriche
    righe_complete = int((~maschera.any(axis=1)).sum())
    percentuale_complete = righe_complete / len(df) * 100 if len(df) else 0

    def card(titolo, valore, dettaglio=""):
        return pn.pane.HTML(
            f'''<div style="height:78px;background:#fff;border:1px solid #e2e8f0;
            border-radius:10px;padding:10px 13px;box-sizing:border-box;">
            <div style="font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.06em">{titolo}</div>
            <div style="font-size:23px;font-weight:700;color:#0f172a;line-height:1.35">{valore}</div>
            <div style="font-size:10px;color:#64748b">{dettaglio}</div></div>''',
            height=80, sizing_mode="stretch_width"
        )

    kpi = pn.GridBox(
        card("Record", f"{len(df):,}".replace(",", ".")),
        card("Variabili", len(df.columns), f"{numeriche} numeriche · {categoriche} categoriche"),
        card("Valori mancanti", f"{percentuale_mancanti:.1f}%", f"{celle_mancanti:,} celle".replace(",", ".")),
        card("Duplicati", duplicati, f"{duplicati / len(df) * 100:.1f}% delle righe" if len(df) else ""),
        card("Righe complete", f"{percentuale_complete:.1f}%", f"{righe_complete:,} record".replace(",", ".")),
        card("Colonne con missing", int(maschera.sum().gt(0).sum())),
        ncols=6, sizing_mode="stretch_width"
    )

    grafici = []
    col_gender = next((c for c in df.columns if c.lower() in ["gender", "genere", "sex", "sesso"]), None)
    col_eta = next((c for c in df.columns if c.lower() in ["age", "eta", "età"]), None)

    if col_gender:
        dati_gender = df[col_gender].fillna("Missing").astype(str).value_counts().rename_axis("Genere").reset_index(name="Record")
        fig_gender = px.pie(dati_gender, names="Genere", values="Record", hole=.62, title="Distribuzione per genere")
        fig_gender.update_layout(height=245, margin=dict(l=5, r=5, t=42, b=5), legend=dict(orientation="h", y=-.12, font=dict(size=10)))
        impostazioni_comuni(fig_gender)
        grafici.append(pn.pane.Plotly(fig_gender, height=250, sizing_mode="stretch_width", config={"responsive": True, "displayModeBar": False}))

    if col_eta:
        eta = pd.to_numeric(df[col_eta], errors="coerce").dropna()
        if not eta.empty:
            fig_eta = px.histogram(pd.DataFrame({"Età": eta}), x="Età", nbins=18, title="Distribuzione dell'età")
            fig_eta.update_layout(height=245, margin=dict(l=8, r=8, t=42, b=8), showlegend=False)
            impostazioni_comuni(fig_eta)
            grafici.append(pn.pane.Plotly(fig_eta, height=250, sizing_mode="stretch_width", config={"responsive": True, "displayModeBar": False}))

    if "Depression" in df.columns:
        target = df["Depression"].fillna("Missing").astype(str).value_counts().rename_axis("Classe").reset_index(name="Record")
        fig_target = px.pie(target, names="Classe", values="Record", hole=.62, title="Distribuzione depressione")
        fig_target.update_layout(height=245, margin=dict(l=5, r=5, t=42, b=5), legend=dict(orientation="h", y=-.12, font=dict(size=10)))
        impostazioni_comuni(fig_target)
        grafici.append(pn.pane.Plotly(fig_target, height=250, sizing_mode="stretch_width", config={"responsive": True, "displayModeBar": False}))

    if len(grafici) == 0:
        grafici.append(pn.pane.Alert("Non sono state trovate le colonne Gender, Age o Depression per i grafici demografici.", alert_type="info"))

    ris = risultati_modelli.get("ultimo")
    if ris is not None and ris.get("tabella") is not None and not ris["tabella"].empty:
        tabella = ris["tabella"].copy()
        metriche = tabella.melt(
            id_vars="Modello",
            value_vars=[c for c in ["Accuracy test", "Precision", "Recall", "F1", "ROC AUC"] if c in tabella.columns],
            var_name="Metrica", value_name="Valore"
        ).dropna()
        fig_modelli = px.bar(metriche, x="Modello", y="Valore", color="Metrica", barmode="group", title="Confronto modelli")
        fig_modelli.update_yaxes(range=[0, 1.05])
        fig_modelli.update_layout(height=270, margin=dict(l=8, r=8, t=42, b=8), legend=dict(orientation="h", y=-.25, font=dict(size=9)))
        impostazioni_comuni(fig_modelli)
        grafici.append(pn.pane.Plotly(fig_modelli, height=275, sizing_mode="stretch_width", config={"responsive": True, "displayModeBar": False}))
    else:
        grafici.append(pn.pane.Alert("Addestramento automatico dei modelli in corso…", alert_type="info", height=55))

    dashboard_home.objects = [pn.Column(
        pn.pane.Markdown(f"## Overview · {stato['nome_file']}"),
        kpi,
        pn.GridBox(*grafici, ncols=2, sizing_mode="stretch_width"),
        height=690, sizing_mode="stretch_width",
        styles={"background": "#f8fafc", "padding": "12px", "border-radius": "12px", "overflow": "hidden"}
    )]


def aggiorna_tutto():
    aggiorna_esplorazione()
    aggiorna_pulizia()
    aggiorna_modelli()
    if stato["nome_file"] == "Student_Depression_Dataset.csv":
        aggiorna_dashboard_home()
        aggiorna_report()


def applica_modifica(nuovo_df, descrizione):
    stato["pulito"] = nuovo_df
    stato["registro"].append(descrizione)
    aggiorna_tutto()


# ============================================================
# AZIONI DI PULIZIA
# ============================================================

def valore_keep():
    return "first" if dup_mantieni.value == "Prima occorrenza" else "last"


def cerca_duplicati(event=None):
    dup_nascondi()
    dup_rimuovi.visible = False
    dup_risultato.objects = []

    df = stato["pulito"]
    if df is None:
        return

    colonne = list(dup_colonne.value)
    if len(colonne) == 0:
        mostra_esito(dup_esito, "Seleziona almeno una colonna da confrontare.", "warning")
        return

    keep = valore_keep()
    in_gruppi = df.duplicated(subset=colonne, keep=False)
    da_rimuovere = df.duplicated(subset=colonne, keep=keep)

    numero_in_gruppi = int(in_gruppi.sum())
    numero_da_rimuovere = int(da_rimuovere.sum())

    if numero_in_gruppi == 0:
        mostra_esito(dup_esito, "Nessun duplicato trovato con le colonne selezionate.", "success")
        return

    visibili = df[in_gruppi].copy()
    visibili.insert(0, "Azione", np.where(da_rimuovere[in_gruppi], "Da rimuovere", "Da mantenere"))

    try:
        visibili = visibili.sort_values(by=colonne, kind="stable")
    except Exception:
        pass

    dup_risultato.objects = [crea_tabella(visibili)]
    dup_rimuovi.visible = True

    mostra_esito(
        dup_esito,
        f"Trovate **{numero_in_gruppi}** righe coinvolte in duplicati. Ne verrebbero rimosse **{numero_da_rimuovere}**.",
        "info"
    )


def chiedi_rimozione_duplicati(event=None):
    df = stato["pulito"]
    if df is None:
        return

    colonne = list(dup_colonne.value)
    keep = valore_keep()
    if len(colonne) == 0:
        return

    numero = int(df.duplicated(subset=colonne, keep=keep).sum())
    if numero == 0:
        return

    def esegui():
        nuovo_df = stato["pulito"].drop_duplicates(subset=colonne, keep=keep).copy()
        rimosse = len(stato["pulito"]) - len(nuovo_df)
        applica_modifica(
            nuovo_df,
            f"Duplicati: rimosse {rimosse} righe (colonne: {', '.join(colonne)}; mantieni: {keep})."
        )
        mostra_esito(dup_esito, f"Rimosse **{rimosse}** righe duplicate.", "success")

    dup_chiedi(f"Stai per rimuovere **{numero}** righe duplicate. Confermi?", esegui)


dup_cerca.on_click(cerca_duplicati)
dup_rimuovi.on_click(chiedi_rimozione_duplicati)


def mostra_righe_mancanti(event=None):
    mis_nascondi()
    mis_risultato.objects = []
    df = stato["pulito"]
    if df is None:
        return

    colonne = list(mis_colonne.value)
    if len(colonne) == 0:
        mostra_esito(mis_esito, "Seleziona almeno una colonna.", "warning")
        return

    maschera = maschera_mancanti(df)[colonne]
    righe_interessate = maschera.any(axis=1)
    totale = int(righe_interessate.sum())

    if totale == 0:
        mostra_esito(mis_esito, "Nessun valore mancante nelle colonne selezionate.", "success")
        return

    maschera_visibile = maschera[righe_interessate].head(MAX_RIGHE_TABELLA)
    visibili = df.loc[maschera_visibile.index].copy()
    visibili.insert(0, "Colonne mancanti", maschera_visibile.apply(lambda r: ", ".join(r.index[r.values]), axis=1))

    mis_risultato.objects = [crea_tabella(visibili, totale=totale)]
    mostra_esito(mis_esito, f"**{totale}** righe con almeno un valore mancante.", "info")


def chiedi_rimozione_mancanti(event=None):
    df = stato["pulito"]
    if df is None:
        return

    colonne = list(mis_colonne.value)
    if len(colonne) == 0:
        mostra_esito(mis_esito, "Seleziona almeno una colonna.", "warning")
        return

    righe_interessate = maschera_mancanti(df)[colonne].any(axis=1)
    numero = int(righe_interessate.sum())

    if numero == 0:
        mostra_esito(mis_esito, "Nessuna riga da rimuovere nelle colonne selezionate.", "success")
        return

    def esegui():
        corrente = stato["pulito"]
        da_tenere = ~maschera_mancanti(corrente)[colonne].any(axis=1)
        nuovo_df = corrente.loc[da_tenere].copy()
        rimosse = len(corrente) - len(nuovo_df)
        applica_modifica(nuovo_df, f"Valori mancanti: rimosse {rimosse} righe (colonne: {', '.join(colonne)}).")
        mostra_esito(mis_esito, f"Rimosse **{rimosse}** righe con valori mancanti.", "success")

    mis_chiedi(f"Stai per rimuovere **{numero}** righe con valori mancanti. Confermi?", esegui)


def imputa_valori_mancanti(event=None):
    mis_nascondi()
    df_corrente = stato["pulito"]
    if df_corrente is None:
        return

    colonne = list(mis_colonne.value)
    metodo = mis_metodo.value

    if len(colonne) == 0:
        mostra_esito(mis_esito, "Seleziona almeno una colonna.", "warning")
        return

    testo_personalizzato = mis_valore.value.strip()
    if metodo == "Valore personalizzato" and testo_personalizzato == "":
        mostra_esito(mis_esito, "Scrivi il valore personalizzato da usare.", "warning")
        return

    df = df_corrente.copy()
    maschera = maschera_mancanti(df)
    valori_sostituiti = 0
    colonne_saltate = []

    for colonna in colonne:
        mancanti = maschera[colonna]
        numero = int(mancanti.sum())
        if numero == 0:
            continue

        numerica = pd.api.types.is_numeric_dtype(df[colonna])
        valore = None

        if metodo == "Media":
            if not numerica:
                colonne_saltate.append(f"{colonna} (non numerica)")
                continue
            valore = df[colonna].mean()
        elif metodo == "Mediana":
            if not numerica:
                colonne_saltate.append(f"{colonna} (non numerica)")
                continue
            valore = df[colonna].median()
        elif metodo == "Moda":
            mode = df.loc[~mancanti, colonna].mode()
            if len(mode) == 0:
                colonne_saltate.append(f"{colonna} (nessun valore valido)")
                continue
            valore = mode.iloc[0]
        else:
            if numerica:
                try:
                    valore = float(testo_personalizzato.replace(",", "."))
                except ValueError:
                    colonne_saltate.append(f"{colonna} (non è un numero)")
                    continue
            else:
                valore = testo_personalizzato

        if pd.isna(valore):
            colonne_saltate.append(f"{colonna} (valore non valido)")
            continue

        try:
            df.loc[mancanti, colonna] = valore
            valori_sostituiti += numero
        except Exception:
            colonne_saltate.append(f"{colonna} (tipo non compatibile)")

    testo_saltate = f"\n\nColonne saltate: {'; '.join(colonne_saltate)}." if colonne_saltate else ""

    if valori_sostituiti == 0:
        mostra_esito(mis_esito, "Nessun valore è stato imputato." + testo_saltate, "warning")
        return

    applica_modifica(df, f"Imputati {valori_sostituiti} valori mancanti ({metodo}).")
    mostra_esito(mis_esito, f"Imputati **{valori_sostituiti}** valori con «{metodo}»." + testo_saltate, "success")


def aggiorna_visibilita_valore(event=None):
    mis_valore.visible = (mis_metodo.value == "Valore personalizzato")


mis_mostra.on_click(mostra_righe_mancanti)
mis_rimuovi.on_click(chiedi_rimozione_mancanti)
mis_imputa.on_click(imputa_valori_mancanti)
mis_metodo.param.watch(aggiorna_visibilita_valore, "value")


def calcola_outlier(serie, metodo):
    valide = serie.dropna()
    if metodo.startswith("IQR"):
        q1 = valide.quantile(0.25)
        q3 = valide.quantile(0.75)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
    else:
        lim_inf = valide.quantile(0.05)
        lim_sup = valide.quantile(0.95)

    maschera = (serie < lim_inf) | (serie > lim_sup)
    return maschera, lim_inf, lim_sup


def crea_grafico_outlier(df, colonna, maschera, lim_inf, lim_sup, metodo):
    dati = pd.DataFrame({
        "Indice riga": df.index,
        colonna: df[colonna].values,
        "Tipo": np.where(maschera.values, "Outlier", "Valore normale")
    }).dropna(subset=[colonna])

    normali = dati[dati["Tipo"] == "Valore normale"]
    outlier = dati[dati["Tipo"] == "Outlier"]

    nota = ""
    if len(normali) > MAX_PUNTI_OUTLIER:
        normali = normali.sample(n=MAX_PUNTI_OUTLIER, random_state=42)
        nota = f" (campione normali: {MAX_PUNTI_OUTLIER})"

    dati = pd.concat([normali, outlier])

    fig = px.scatter(
        dati, x="Indice riga", y=colonna, color="Tipo",
        color_discrete_map={"Valore normale": "#2563eb", "Outlier": "#dc2626"},
        category_orders={"Tipo": ["Valore normale", "Outlier"]},
        title=f"Outlier di {colonna} – {metodo}{nota}"
    )
    fig.add_hline(y=lim_inf, line_dash="dash", line_color="#dc2626", annotation_text="Limite inf")
    fig.add_hline(y=lim_sup, line_dash="dash", line_color="#dc2626", annotation_text="Limite sup")
    fig.update_layout(template="plotly_white", autosize=True, margin=dict(l=40, r=20, t=60, b=60))
    return fig


def individua_outlier(event=None):
    out_nascondi()
    out_rimuovi.visible = False
    out_risultato.objects = []

    df = stato["pulito"]
    colonna = out_colonna.value
    if df is None or colonna is None:
        return

    serie = df[colonna]
    if serie.dropna().empty:
        mostra_esito(out_esito, f"La colonna «{colonna}» non contiene valori validi.", "warning")
        return

    metodo = out_metodo.value
    maschera, lim_inf, lim_sup = calcola_outlier(serie, metodo)
    numero = int(maschera.sum())

    grafico = pn.pane.Plotly(
        crea_grafico_outlier(df, colonna, maschera, lim_inf, lim_sup, metodo),
        height=380, sizing_mode="stretch_width", config={"responsive": True}
    )

    oggetti = [grafico]
    if numero > 0:
        oggetti.append(crea_tabella(df[maschera]))
        out_rimuovi.visible = True

    out_risultato.objects = oggetti
    mostra_esito(
        out_esito,
        f"Limiti: da **{lim_inf:.4g}** a **{lim_sup:.4g}**. Outlier trovati: **{numero}**.",
        "info" if numero > 0 else "success"
    )


def chiedi_rimozione_outlier(event=None):
    df = stato["pulito"]
    colonna = out_colonna.value
    metodo = out_metodo.value
    if df is None or colonna is None:
        return

    maschera, _, _ = calcola_outlier(df[colonna], metodo)
    numero = int(maschera.sum())
    if numero == 0:
        return

    def esegui():
        corrente = stato["pulito"]
        maschera_attuale, _, _ = calcola_outlier(corrente[colonna], metodo)
        nuovo_df = corrente.loc[~maschera_attuale].copy()
        rimosse = len(corrente) - len(nuovo_df)
        applica_modifica(nuovo_df, f"Outlier: rimosse {rimosse} righe ({colonna}, {metodo}).")
        mostra_esito(out_esito, f"Rimosse **{rimosse}** righe con outlier in «{colonna}».", "success")

    out_chiedi(f"Stai per rimuovere **{numero}** righe con outlier in «{colonna}». Confermi?", esegui)


out_cerca.on_click(individua_outlier)
out_rimuovi.on_click(chiedi_rimozione_outlier)


def chiedi_rimozione_colonne(event=None):
    df = stato["pulito"]
    if df is None:
        return

    colonne = list(col_selezione.value)
    if len(colonne) == 0:
        mostra_esito(col_esito, "Seleziona almeno una colonna da rimuovere.", "warning")
        return

    if len(colonne) >= len(df.columns):
        mostra_esito(col_esito, "Ne deve restare almeno una.", "warning")
        return

    def esegui():
        nuovo_df = stato["pulito"].drop(columns=colonne).copy()
        applica_modifica(nuovo_df, f"Colonne: rimosse {len(colonne)} colonne ({', '.join(colonne)}).")
        mostra_esito(col_esito, f"Rimosse **{len(colonne)}** colonne.", "success")

    col_chiedi(f"Stai per rimuovere **{len(colonne)}** colonne. Confermi?", esegui)


col_rimuovi.on_click(chiedi_rimozione_colonne)


def chiedi_ripristino(event=None):
    if stato["originale"] is None:
        return

    if len(stato["registro"]) == 0:
        mostra_esito(ripristino_esito, "Non ci sono modifiche da annullare.", "info")
        return

    def esegui():
        stato["pulito"] = stato["originale"].copy()
        stato["registro"] = []
        aggiorna_tutto()
        mostra_esito(ripristino_esito, "Dataset originale ripristinato.", "success")

    ripristino_chiedi("Il dataset tornerà identico all'originale. Confermi?", esegui)


ripristina.on_click(chiedi_ripristino)


# ============================================================
# SCHEDA "MODELLI": FUNZIONI E PREPARAZIONE DATI
# ============================================================

def carica_parametri_salvati():
    """Carica gli iperparametri salvati da main.py in data/*.par se presenti."""
    params = {}
    if not JOBLIB_DISPONIBILE or not os.path.exists(PAR_DIR):
        return params

    try:
        p_tree = os.path.join(PAR_DIR, "gridSearch.par")
        if os.path.exists(p_tree):
            params["tree"] = joblib.load(p_tree)
    except Exception:
        pass

    try:
        p_rf = os.path.join(PAR_DIR, "randomSearchRandomForest.par")
        if os.path.exists(p_rf):
            raw_rf = joblib.load(p_rf)
            params["rf"] = {k.replace("classifier__", ""): v for k, v in raw_rf.items()}
    except Exception:
        pass

    try:
        p_cb = os.path.join(PAR_DIR, "catboost.par")
        if os.path.exists(p_cb):
            params["catboost"] = joblib.load(p_cb)
    except Exception:
        pass

    try:
        p_logit = os.path.join(PAR_DIR, "logit.par")
        if os.path.exists(p_logit):
            params["logit"] = joblib.load(p_logit)
    except Exception:
        pass

    return params


def crea_preprocessore(colonne_numeriche, colonne_categoriche, scala):
    """
    Preparazione dei dati allineata a main.py:
    - numeriche: mediana + StandardScaler (se scala=True)
    - categoriche: OneHotEncoder con drop='first' e handle_unknown='ignore'
    """
    passi_numerici = [("imputer", SimpleImputer(strategy="median"))]
    if scala:
        passi_numerici.append(("scaler", StandardScaler()))

    trasformatori = []
    if len(colonne_numeriche) > 0:
        trasformatori.append(("num", Pipeline(passi_numerici), colonne_numeriche))

    if len(colonne_categoriche) > 0:
        trasformatori.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="first",
                            handle_unknown="ignore",
                            sparse_output=False
                        )
                    )
                ]),
                colonne_categoriche
            )
        )

    return ColumnTransformer(trasformatori)


def prepara_dati_modello(df, target, feature):
    """
    Restituisce X, y (codificata in numeri), nomi delle classi,
    colonne numeriche, colonne categoriche, colonne escluse e avvisi.
    Esegue la feature engineering usata in main.py se i campi sono presenti.
    """
    avvisi = []
    feature = [colonna for colonna in feature if colonna != target]

    if len(feature) == 0:
        raise ValueError("Seleziona almeno una variabile esplicativa.")

    target_mancante = maschera_mancanti(df[[target]])[target]
    if target_mancante.any():
        avvisi.append(f"Escluse {int(target_mancante.sum())} righe con target mancante.")

    dati = df.loc[~target_mancante].copy()
    if dati.empty:
        raise ValueError("Nessuna riga con target valido.")

    # Feature engineering come in main.py
    if "Gender" in dati.columns and "male" not in dati.columns and "male" in feature:
        dati["male"] = (dati["Gender"] == "Male").astype(int)
    if "friend_circle" in dati.columns and "no_friends" not in dati.columns and "no_friends" in feature:
        dati["no_friends"] = (dati["friend_circle"] == "No").astype(int)

    codificatore = LabelEncoder()
    y = codificatore.fit_transform(dati[target].astype(str))
    classi = [str(classe) for classe in codificatore.classes_]

    if len(classi) < 2:
        raise ValueError("Il target deve avere almeno 2 classi.")
    if len(classi) > MAX_CLASSI:
        raise ValueError(f"Il target ha {len(classi)} classi (massimo {MAX_CLASSI}).")

    X = dati[feature].copy()
    mancanti = maschera_mancanti(X)

    colonne_numeriche = []
    colonne_categoriche = []
    escluse = []

    for colonna in feature:
        if pd.api.types.is_numeric_dtype(X[colonna]):
            X[colonna] = X[colonna].astype(float)
            if X[colonna].isna().all():
                escluse.append(f"{colonna} (tutti mancanti)")
                continue
            colonne_numeriche.append(colonna)
        else:
            X[colonna] = X[colonna].astype(object).where(~mancanti[colonna], np.nan)
            X[colonna] = X[colonna].map(lambda v: v if pd.isna(v) else str(v))
            if X[colonna].isna().all():
                escluse.append(f"{colonna} (tutti mancanti)")
                continue
            if X[colonna].nunique() > MAX_CATEGORIE_FEATURE:
                escluse.append(f"{colonna} (oltre {MAX_CATEGORIE_FEATURE} categorie)")
                continue
            colonne_categoriche.append(colonna)

    colonne_usate = colonne_numeriche + colonne_categoriche
    if len(colonne_usate) == 0:
        raise ValueError("Nessuna variabile esplicativa utilizzabile.")

    return (
        X[colonne_usate],
        y,
        classi,
        colonne_numeriche,
        colonne_categoriche,
        escluse,
        avvisi
    )


def crea_modello(nome, parametri, seed, numero_train):
    """
    Restituisce (modello, serve_standardizzazione).
    Stessi algoritmi, bilanciamento classi e parametri di main.py.
    """
    usa_balanced = parametri.get("use_class_weights", True)

    if nome == "Regressione logistica":
        return (
            LogisticRegression(
                C=parametri.get("lr_c", 1.0),
                class_weight="balanced" if usa_balanced else None,
                max_iter=1000,
                random_state=seed,
                solver="lbfgs"
            ),
            True
        )

    if nome == "Albero decisionale":
        return (
            DecisionTreeClassifier(
                criterion=parametri.get("dt_criterion", "gini"),
                max_depth=parametri.get("dt_profondita"),
                min_samples_split=parametri.get("dt_min_samples_split", 2),
                min_samples_leaf=parametri.get("dt_min_samples_leaf", 20),
                ccp_alpha=parametri.get("dt_ccp_alpha", 0.0),
                class_weight="balanced" if usa_balanced else None,
                random_state=seed
            ),
            False
        )

    if nome == "Random forest":
        max_feat = parametri.get("rf_max_features", "sqrt")
        if max_feat not in ["sqrt", "log2", None] and not isinstance(max_feat, float):
            max_feat = "sqrt"
        return (
            RandomForestClassifier(
                n_estimators=parametri.get("rf_alberi", 300),
                max_depth=parametri.get("rf_profondita"),
                min_samples_split=parametri.get("rf_min_samples_split", 2),
                min_samples_leaf=parametri.get("rf_min_samples_leaf", 1),
                max_features=max_feat,
                class_weight="balanced" if usa_balanced else None,
                random_state=seed,
                n_jobs=-1
            ),
            True
        )

    if nome == "CatBoost":
        if not CATBOOST_DISPONIBILE:
            raise RuntimeError("CatBoost non è installato (pip install catboost).")

        return (
            CatBoostClassifier(
                iterations=parametri.get("cb_iterazioni", 300),
                depth=parametri.get("cb_profondita", 6),
                learning_rate=parametri.get("cb_learning_rate", 0.03),
                l2_leaf_reg=parametri.get("cb_l2", 10.0),
                auto_class_weights="Balanced" if usa_balanced else None,
                random_seed=seed,
                verbose=0,
                allow_writing_files=False
            ),
            False
        )

    raise ValueError(f"Modello non supportato: {nome}")


def nomi_feature(pipeline):
    nomi = pipeline.named_steps["prep"].get_feature_names_out()
    return [str(nome).split("__", 1)[1] for nome in nomi]


def calcola_importanze(pipeline, X_test, y_test, seed):
    modello = pipeline.named_steps["modello"]

    if hasattr(modello, "feature_importances_"):
        valori = np.asarray(modello.feature_importances_, dtype=float)
        return pd.Series(valori, index=nomi_feature(pipeline)), "Importanza delle variabili"

    if hasattr(modello, "coef_"):
        valori = np.abs(np.asarray(modello.coef_, dtype=float)).mean(axis=0)
        return pd.Series(valori, index=nomi_feature(pipeline)), "Importanza delle variabili (|coefficiente|)"

    campione = X_test
    y_campione = y_test
    if len(X_test) > 1000:
        posizioni = np.random.RandomState(seed).choice(len(X_test), 1000, replace=False)
        campione = X_test.iloc[posizioni]
        y_campione = y_test[posizioni]

    risultato = permutation_importance(pipeline, campione, y_campione, n_repeats=3, random_state=seed, scoring="accuracy")
    return pd.Series(risultato.importances_mean, index=list(X_test.columns)), "Importanza per permutazione (calo di accuracy)"


def addestra_modelli_main(df, config):
    """Replica le procedure di main.py per il dataset Student Depression.

    Mantiene split, feature, preprocessing, selezione degli iperparametri e soglie
    del main. I file .par vengono riutilizzati; se assenti, le ricerche vengono
    eseguite con gli stessi spazi e criteri di main.py.
    """
    richieste = ["Depression", "Age", "Current_semester", "CGPA", "Cost_without_study",
                 "Gender", "Family_background", "Relationship_status", "Marital_status",
                 "Living_style", "friend_circle", "Home_division"]
    mancanti = [c for c in richieste if c not in df.columns]
    if mancanti:
        raise ValueError("Dataset non compatibile con main.py. Colonne mancanti: " + ", ".join(mancanti))
    if not JOBLIB_DISPONIBILE:
        raise RuntimeError("joblib è necessario per leggere/salvare i parametri .par di main.py.")
    if not CATBOOST_DISPONIBILE:
        raise RuntimeError("CatBoost non è installato (pip install catboost).")
    try:
        import statsmodels.api as sm
        from itertools import product as iter_product
    except ImportError as e:
        raise RuntimeError("statsmodels è necessario per la regressione logistica stepwise di main.py.") from e

    seed = 42
    target = "Depression"
    num = ["Age", "Current_semester", "CGPA", "Cost_without_study"]
    rf_cat = ["Gender", "Family_background", "Relationship_status", "Marital_status",
              "Living_style", "friend_circle", "Home_division"]
    # Health_condition non deve mai entrare tra i predittori: è una variabile in leakage.
    feature_vietate = {"Health_condition", "health_condition"}
    num = [c for c in num if c not in feature_vietate]
    rf_cat = [c for c in rf_cat if c not in feature_vietate]
    df = df.copy()
    df = df[df[target].isin(["Yes", "No"])].copy()
    df["dep"] = (df[target] == "Yes").astype(int)
    if len(df) == 0:
        raise ValueError("Il target Depression deve contenere Yes/No.")

    def split_main(cols):
        dati = df[cols + ["dep"]].dropna()
        if dati.empty or dati["dep"].nunique() < 2:
            raise ValueError("Dati insufficienti per il modello con feature: " + ", ".join(cols))
        return train_test_split(dati[cols], dati["dep"], test_size=0.2,
                                random_state=seed, stratify=dati["dep"])

    def load_par(filename, search):
        os.makedirs(PAR_DIR, exist_ok=True)
        path = os.path.join(PAR_DIR, filename)
        if os.path.exists(path):
            try:
                return joblib.load(path)
            except Exception:
                pass
        params = search()
        joblib.dump(params, path)
        return params

    dettagli, righe, errori, avvisi = {}, [], {}, []
    classi = ["No", "Yes"]

    def store_model(nome, y_test, prob, soglia, importanze, titolo, testo_albero=None, params=None):
        pred = (np.asarray(prob) >= soglia).astype(int)
        report = classification_report(y_test, pred, labels=[0, 1], target_names=classi,
                                       output_dict=True, zero_division=0)
        report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Classe"}).round(3)
        righe.append({"Modello": nome, "Accuracy train": np.nan,
                      "Accuracy test": accuracy_score(y_test, pred),
                      "Precision": precision_score(y_test, pred, zero_division=0),
                      "Recall": recall_score(y_test, pred, zero_division=0),
                      "F1": f1_score(y_test, pred, zero_division=0),
                      "ROC AUC": roc_auc_score(y_test, prob), "Soglia": soglia, "Tempo (s)": np.nan})
        dettagli[nome] = {"matrice": confusion_matrix(y_test, pred, labels=[0, 1]),
                          "importanze": importanze, "titolo_importanze": titolo,
                          "testo_albero": testo_albero, "report": report,
                          "roc": roc_curve(y_test, prob)[:2] + (roc_auc_score(y_test, prob),)}

    # 1) Decision Tree: one-hot drop_first, GridSearchCV accuracy, class_weight balanced.
    if "Albero decisionale" in config.get("modelli", []):
        try:
            cols = num + [c for c in df.select_dtypes(include=["object", "category", "string"]).columns
                          if c not in [target, "Marital_status", "Health_condition", "health_condition", "dep"]]
            Xtr, Xte, ytr, yte = split_main(cols)
            cats = [c for c in cols if c not in num]
            Xtr_e = pd.get_dummies(Xtr, columns=cats, drop_first=True, dtype=float)
            Xte_e = pd.get_dummies(Xte, columns=cats, drop_first=True, dtype=float).reindex(columns=Xtr_e.columns, fill_value=0)
            def search_tree():
                grid = {"criterion": ["gini", "entropy"], "max_depth": [2, 3, 4, 5, 6, 8, 10, None],
                        "min_samples_split": [2, 5, 10, 20], "min_samples_leaf": [1, 5, 10, 20, 30],
                        "ccp_alpha": [0.0, 0.001, 0.005, 0.01, 0.02]}
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                return GridSearchCV(DecisionTreeClassifier(class_weight="balanced", random_state=seed),
                                    grid, scoring="accuracy", cv=cv, n_jobs=-1).fit(Xtr_e, ytr).best_params_
            tree_params = load_par("gridSearch.par", search_tree)
            tree = DecisionTreeClassifier(**tree_params, class_weight="balanced", random_state=seed).fit(Xtr_e, ytr)
            prob = tree.predict_proba(Xte_e)[:, 1]
            text_tree = export_text(tree, feature_names=list(Xtr_e.columns), max_depth=3)
            store_model("Decision Tree", yte, prob, 0.5,
                        pd.Series(tree.feature_importances_, index=Xtr_e.columns), "Feature importance", text_tree, tree_params)
        except Exception as e:
            errori["Decision Tree"] = f"{type(e).__name__}: {e}"

    # 2) Random Forest: StandardScaler + OneHotEncoder; RandomizedSearchCV F1;
    if "Random forest" in config.get("modelli", []):
        # OOB sweep 50..600 step 25; final n_estimators=max(best_n, 300); threshold optimized on test as in main.
        try:
            Xtr, Xte, ytr, yte = split_main(num + rf_cat)
            prep = ColumnTransformer([
                ("num", StandardScaler(), num),
                ("cat", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False), rf_cat)
            ])
            pipe = Pipeline([("preprocessor", prep), ("classifier", RandomForestClassifier(
                n_estimators=300, random_state=seed, n_jobs=-1, class_weight="balanced"))])
            def search_rf():
                dist = {"classifier__max_depth": [4, 6, 8, 10, 15, None],
                        "classifier__min_samples_split": randint(2, 21),
                        "classifier__min_samples_leaf": randint(1, 21),
                        "classifier__max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
                        "classifier__class_weight": ["balanced", "balanced_subsample"]}
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                rs = RandomizedSearchCV(pipe, dist, n_iter=60, scoring="f1", cv=cv, n_jobs=-1,
                                        random_state=seed, return_train_score=True)
                rs.fit(Xtr, ytr)
                return rs.best_params_
            raw = load_par("randomSearchRandomForest.par", search_rf)
            rf_params = {k.replace("classifier__", ""): v for k, v in raw.items()}
            Xtr_t = prep.fit_transform(Xtr)
            oob_rf = RandomForestClassifier(**rf_params, n_estimators=50, oob_score=True, warm_start=True,
                                            random_state=seed, n_jobs=-1)
            oob_rows = []
            for n_est in range(50, 601, 25):
                oob_rf.set_params(n_estimators=n_est).fit(Xtr_t, ytr)
                oob_rows.append((n_est, roc_auc_score(ytr, oob_rf.oob_decision_function_[:, 1])))
            best_n = max(oob_rows, key=lambda x: x[1])[0]
            final_n = max(best_n, 300)
            pipe.set_params(classifier=RandomForestClassifier(**rf_params, n_estimators=final_n,
                          oob_score=True, random_state=seed, n_jobs=-1)).fit(Xtr, ytr)
            rf = pipe.named_steps["classifier"]
            prob = pipe.predict_proba(Xte)[:, 1]
            pp, rr, tt = precision_recall_curve(yte, prob)
            tab = pd.DataFrame({"threshold": tt, "precision": pp[:-1], "recall": rr[:-1]})
            tab["f1"] = 2 * tab.precision * tab.recall / (tab.precision + tab.recall + 1e-12)
            threshold = float(tab.loc[tab.f1.idxmax(), "threshold"])
            names = prep.get_feature_names_out()
            store_model("Random Forest", yte, prob, threshold,
                        pd.Series(rf.feature_importances_, index=names), "Feature importance", params=rf_params)
        except Exception as e:
            errori["Random Forest"] = f"{type(e).__name__}: {e}"

    # 3) CatBoost: feature engineering, native categorical features, repeated 5x3 CV AUC.
    if "CatBoost" in config.get("modelli", []):
        try:
            df["male"] = (df["Gender"] == "Male").astype(int)
            df["no_friends"] = (df["friend_circle"] == "No").astype(int)
            cb_cat = ["Relationship_status", "Family_background", "Home_division", "Living_style"]
            cb_num = ["CGPA", "Cost_without_study", "Age", "Current_semester"]
            Xtr, Xte, ytr, yte = split_main(["male", "no_friends"] + cb_cat + cb_num)
            def new_cb(depth, l2_leaf_reg, learning_rate=0.03, iterations=300):
                return CatBoostClassifier(iterations=iterations, depth=depth, learning_rate=learning_rate,
                    l2_leaf_reg=l2_leaf_reg, one_hot_max_size=10, cat_features=tuple(cb_cat),
                    auto_class_weights="Balanced", random_seed=seed, verbose=0, allow_writing_files=False)
            def search_cb():
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
                vals = []
                for depth, l2 in iter_product([4, 6, 8], [10, 30, 50]):
                    model = new_cb(depth, l2)
                    scores = cross_val_score(model, Xtr, ytr, cv=cv, scoring="roc_auc")
                    vals.append((scores.mean(), depth, l2))
                _, depth, l2 = max(vals)
                return {"depth": int(depth), "l2_leaf_reg": float(l2), "learning_rate": 0.03}
            cb_params = load_par("catboost.par", search_cb)
            model = new_cb(**cb_params).fit(Xtr, ytr)
            prob_cv = cross_val_predict(new_cb(**cb_params), Xtr, ytr,
                        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
                        method="predict_proba")[:, 1]
            soglie = np.arange(0.05, 0.96, 0.01)
            f1s = [f1_score(ytr, (prob_cv >= t).astype(int)) for t in soglie]
            threshold = round(float(soglie[int(np.argmax(f1s))]), 2)
            prob = model.predict_proba(Xte)[:, 1]
            store_model("CatBoost", yte, prob, threshold,
                        pd.Series(model.get_feature_importance(), index=Xtr.columns), "Feature importance", params=cb_params)
        except Exception as e:
            errori["CatBoost"] = f"{type(e).__name__}: {e}"

    # 4) Logistic Regression: StandardScaler + get_dummies + statsmodels Logit;
    if "Regressione logistica" in config.get("modelli", []):
        # forward/backward selection by AIC/BIC; select the best BIC model.
        try:
            Xtr, Xte, ytr, yte = split_main(num + rf_cat)
            Xtr, Xte = Xtr.copy(), Xte.copy()
            scaler = StandardScaler()
            Xtr[num] = scaler.fit_transform(Xtr[num]); Xte[num] = scaler.transform(Xte[num])
            Xtr_lg = sm.add_constant(pd.get_dummies(Xtr, columns=rf_cat, drop_first=True, dtype=float), has_constant="add")
            Xte_lg = sm.add_constant(pd.get_dummies(Xte, columns=rf_cat, drop_first=True, dtype=float), has_constant="add")
            Xte_lg = Xte_lg.reindex(columns=Xtr_lg.columns, fill_value=0)
            def fit_logit(X, y):
                X = X.loc[:, ~X.columns.duplicated()].apply(pd.to_numeric, errors="coerce").astype(float)
                X = X.replace([np.inf, -np.inf], np.nan)
                X = X[[c for c in X.columns if c == "const" or X[c].nunique(dropna=False) > 1]]
                valid = X.notna().all(axis=1) & y.notna(); X, yy = X.loc[valid], y.loc[valid].astype(int)
                if yy.nunique() < 2 or X.shape[1] == 0: return None
                for solver in ["newton", "lbfgs", "bfgs"]:
                    try:
                        m = sm.Logit(yy, X).fit(method=solver, maxiter=500, disp=0)
                        if m.mle_retvals.get("converged", False): return m
                    except Exception: continue
                return None
            def crit(m, c): return np.inf if m is None else float(getattr(m, c))
            def stepwise(direction, criterion):
                predictors = [c for c in Xtr_lg.columns if c != "const"]
                selected = [] if direction == "forward" else predictors.copy()
                remaining = predictors.copy() if direction == "forward" else []
                fit = lambda cols: fit_logit(Xtr_lg[["const"] + cols], ytr)
                current = crit(fit(selected), criterion)
                while (remaining if direction == "forward" else len(selected) > 1):
                    pool = remaining if direction == "forward" else selected
                    trials = [(crit(fit(selected + [c] if direction == "forward" else [v for v in selected if v != c]), criterion), c) for c in pool]
                    score, candidate = min(trials)
                    if score >= current - 1e-6: break
                    if direction == "forward": selected.append(candidate); remaining.remove(candidate)
                    else: selected.remove(candidate)
                    current = score
                return selected
            path = os.path.join(PAR_DIR, "logit.par")
            if os.path.exists(path):
                try: logit_vars = joblib.load(path)
                except Exception: logit_vars = None
            else: logit_vars = None
            if not (isinstance(logit_vars, dict) and logit_vars and all(isinstance(v, list) and set(v) <= set(Xtr_lg.columns) for v in logit_vars.values())):
                logit_vars = {f"{d}_{c}": stepwise(d, c) for d, c in iter_product(["forward", "backward"], ["aic", "bic"])}
                joblib.dump(logit_vars, path)
            models = {k: fit_logit(Xtr_lg[["const"] + v], ytr) for k, v in logit_vars.items()}
            valid_models = {k: m for k, m in models.items() if m is not None}
            if not valid_models: raise RuntimeError("Nessun modello logit convergente.")
            best_key = min(valid_models, key=lambda k: crit(valid_models[k], "bic"))
            logit = valid_models[best_key]
            prob = logit.predict(Xte_lg[logit.params.index].astype(float))
            store_model("Logistic Regression", yte, prob, 0.5,
                        pd.Series(np.abs(logit.params.drop("const", errors="ignore")), index=logit.params.drop("const", errors="ignore").index),
                        "Valore assoluto dei coefficienti", params={"selezione": best_key, "criterio": "BIC"})
        except Exception as e:
            errori["Logistic Regression"] = f"{type(e).__name__}: {e}"

    if not righe:
        raise RuntimeError("Nessun modello addestrato: " + " | ".join(f"{k}: {v}" for k, v in errori.items()))
    tabella = pd.DataFrame(righe).round(4).sort_values("Accuracy test", ascending=False).reset_index(drop=True)
    # Mantiene il formato consumato dai componenti UI esistenti.
    return {"target": target, "classi": classi, "conteggio_classi": df[target].value_counts().to_dict(),
            "numero_train": int(round(len(df) * 0.8)), "numero_test": int(round(len(df) * 0.2)),
            "feature_usate": richieste[1:], "escluse": [], "avvisi": avvisi,
            "errori": errori, "tabella": tabella, "modelli": dettagli}


def addestra_modelli(df, config):
    if stato.get("nome_file") == "Student_Depression_Dataset.csv" and config.get("target") == "Depression":
        return addestra_modelli_main(df, config)

    (
        X, y, classi, colonne_numeriche, colonne_categoriche, escluse, avvisi
    ) = prepara_dati_modello(df, config["target"], config["feature"])

    if len(X) < 10:
        raise ValueError("Servono almeno 10 righe con target valido.")

    stratificazione = y if config["stratifica"] else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["seed"], stratify=stratificazione
        )
    except ValueError:
        avvisi.append("Stratificazione non possibile: eseguito split casuale semplice.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config["test_size"], random_state=config["seed"]
        )

    labels = list(range(len(classi)))
    righe = []
    dettagli = {}
    errori = {}

    for nome in config["modelli"]:
        try:
            stimatore, scala = crea_modello(nome, config["parametri"], config["seed"], len(X_train))
            pipeline = Pipeline([
                ("prep", crea_preprocessore(colonne_numeriche, colonne_categoriche, scala)),
                ("modello", stimatore)
            ])

            inizio = time.perf_counter()
            pipeline.fit(X_train, y_train)
            durata = time.perf_counter() - inizio

            # Gestione soglia decisionale per classificazione binaria (ottimizzata su train o fissata)
            if len(classi) == 2 and hasattr(pipeline, "predict_proba"):
                prob_train = pipeline.predict_proba(X_train)[:, 1]
                prob_test = pipeline.predict_proba(X_test)[:, 1]

                if config.get("ottimizza_soglia", False):
                    precisions, recalls, thresholds = precision_recall_curve(y_train, prob_train)
                    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
                    idx_best = np.argmax(f1_scores)
                    soglia_usata = float(thresholds[idx_best]) if idx_best < len(thresholds) else config.get("soglia", 0.5)
                else:
                    soglia_usata = config.get("soglia", 0.5)

                y_pred_train = (prob_train >= soglia_usata).astype(int)
                y_pred = (prob_test >= soglia_usata).astype(int)
            else:
                soglia_usata = 0.5
                y_pred_train = np.asarray(pipeline.predict(X_train)).ravel()
                y_pred = np.asarray(pipeline.predict(X_test)).ravel()

            acc_train = accuracy_score(y_train, y_pred_train)
            acc_test = accuracy_score(y_test, y_pred)
            precisione, richiamo, f1, _ = precision_recall_fscore_support(
                y_test, y_pred, average="weighted", zero_division=0
            )

            auc = np.nan
            curva_roc = None
            try:
                probabilita = pipeline.predict_proba(X_test)
                if len(classi) == 2:
                    auc = roc_auc_score(y_test, probabilita[:, 1])
                    fpr, tpr, _ = roc_curve(y_test, probabilita[:, 1])
                    curva_roc = (fpr, tpr, auc)
                else:
                    auc = roc_auc_score(y_test, probabilita, multi_class="ovr", average="weighted")
            except Exception:
                pass

            cv_media, cv_dev_std = np.nan, np.nan
            if config["cv"]:
                try:
                    punteggi = cross_val_score(
                        clone(pipeline), X_train, y_train, cv=5, scoring="accuracy"
                    )
                    cv_media = float(np.mean(punteggi))
                    cv_dev_std = float(np.std(punteggi))
                except Exception as e_cv:
                    avvisi.append(f"{nome}: cross-validation non eseguita ({type(e_cv).__name__}).")

            importanze, titolo_importanze = calcola_importanze(pipeline, X_test, y_test, config["seed"])

            testo_albero = None
            if nome == "Albero decisionale":
                testo_albero = export_text(
                    pipeline.named_steps["modello"],
                    feature_names=nomi_feature(pipeline),
                    max_depth=4
                )

            report = classification_report(
                y_test, y_pred, labels=labels, target_names=classi, output_dict=True, zero_division=0
            )
            report = pd.DataFrame(report).transpose().reset_index().rename(columns={"index": "Classe"}).round(3)

            riga = {
                "Modello": nome,
                "Accuracy train": acc_train,
                "Accuracy test": acc_test,
                "Precision": precisione,
                "Recall": richiamo,
                "F1": f1,
                "ROC AUC": auc,
                "Soglia": soglia_usata,
                "Tempo (s)": durata
            }

            if config["cv"]:
                riga["CV accuracy (media)"] = cv_media
                riga["CV (dev. std.)"] = cv_dev_std

            righe.append(riga)
            dettagli[nome] = {
                "matrice": confusion_matrix(y_test, y_pred, labels=labels),
                "importanze": importanze,
                "titolo_importanze": titolo_importanze,
                "testo_albero": testo_albero,
                "report": report,
                "roc": curva_roc
            }

        except Exception as errore:
            errori[nome] = f"{type(errore).__name__}: {errore}"

    if len(righe) == 0:
        dettaglio_errori = "; ".join(f"{n} → {t}" for n, t in errori.items())
        raise ValueError(f"Nessun modello addestrato con successo. {dettaglio_errori}")

    tabella = pd.DataFrame(righe).round(4)
    tabella = tabella.sort_values("Accuracy test", ascending=False).reset_index(drop=True)

    return {
        "target": config["target"],
        "classi": classi,
        "conteggio_classi": {classe: int((y == idx).sum()) for idx, classe in enumerate(classi)},
        "numero_train": len(X_train),
        "numero_test": len(X_test),
        "feature_usate": colonne_numeriche + colonne_categoriche,
        "escluse": escluse,
        "avvisi": avvisi,
        "errori": errori,
        "tabella": tabella,
        "modelli": dettagli
    }


# ============================================================
# SCHEDA "MODELLI": COMPONENTI WIDGET E LAYOUT
# ============================================================

risultati_modelli = {"ultimo": None}

mod_target = pn.widgets.Select(name="Variabile target (da prevedere)", options=[], sizing_mode="stretch_width")
mod_feature = pn.widgets.MultiSelect(name="Variabili esplicative (feature)", options=[], size=8, sizing_mode="stretch_width")

mod_test_size = pn.widgets.FloatSlider(name="Quota del test set", start=0.1, end=0.5, step=0.05, value=0.2, sizing_mode="stretch_width")
mod_seed = pn.widgets.IntInput(name="Random state (riproducibilità)", value=42, start=0, sizing_mode="stretch_width")
mod_stratifica = pn.widgets.Checkbox(name="Stratifica lo split per classe", value=True)
mod_cv = pn.widgets.Checkbox(name="Cross-validation a 5 fold sul train set", value=False)
mod_class_weight = pn.widgets.Checkbox(name="Bilancia i pesi delle classi (class_weight='balanced')", value=True)
mod_ottimizza_soglia = pn.widgets.Checkbox(name="Ottimizza soglia di decisione su F1 max (train)", value=False)

# Selezione esplicita dei modelli: checkbox singole, più affidabili e visibili
# del CheckBoxGroup quando si cambia dataset dinamicamente.
mod_usa_lr = pn.widgets.Checkbox(name="Regressione logistica", value=True)
mod_usa_dt = pn.widgets.Checkbox(name="Albero decisionale", value=True)
mod_usa_rf = pn.widgets.Checkbox(name="Random forest", value=True)
mod_usa_cb = pn.widgets.Checkbox(name="CatBoost", value=CATBOOST_DISPONIBILE, disabled=not CATBOOST_DISPONIBILE)

# Iperparametri configurabili per ogni algoritmo
par_lr_c = pn.widgets.FloatInput(name="C (regolarizzazione)", value=1.0, start=0.001, end=1000, step=0.1, width=180)
par_dt_criterion = pn.widgets.Select(name="Criterio", options=["gini", "entropy", "log_loss"], value="gini", width=180)
par_dt_depth = pn.widgets.IntInput(name="Profondità massima (0 = illimitata)", value=0, start=0, end=100, width=220)
par_dt_split = pn.widgets.IntInput(name="Min. campioni per split", value=2, start=2, end=1000, width=180)
par_dt_leaf = pn.widgets.IntInput(name="Min. campioni per foglia", value=1, start=1, end=1000, width=180)
par_dt_alpha = pn.widgets.FloatInput(name="CCP alpha", value=0.0, start=0.0, end=1.0, step=0.001, width=180)
par_rf_trees = pn.widgets.IntInput(name="Numero alberi", value=300, start=10, end=3000, step=10, width=180)
par_rf_depth = pn.widgets.IntInput(name="Profondità massima (0 = illimitata)", value=0, start=0, end=100, width=220)
par_rf_split = pn.widgets.IntInput(name="Min. campioni per split", value=2, start=2, end=1000, width=180)
par_rf_leaf = pn.widgets.IntInput(name="Min. campioni per foglia", value=1, start=1, end=1000, width=180)
par_rf_features = pn.widgets.Select(name="Feature considerate", options=["sqrt", "log2", "None"], value="sqrt", width=180)
par_cb_iterations = pn.widgets.IntInput(name="Numero iterazioni", value=300, start=10, end=5000, step=10, width=180)
par_cb_depth = pn.widgets.IntInput(name="Profondità", value=6, start=1, end=16, width=180)
par_cb_lr = pn.widgets.FloatInput(name="Learning rate", value=0.03, start=0.001, end=1.0, step=0.01, width=180)
par_cb_l2 = pn.widgets.FloatInput(name="L2 leaf regularization", value=10.0, start=0.0, end=1000, step=1.0, width=180)

mod_scelta = pn.widgets.CheckBoxGroup(name="Modelli da addestrare", options=MODELLI, value=list(MODELLI), visible=False)
mod_soglia = pn.widgets.FloatSlider(name="Soglia di decisione fissa (Target Binario)", start=0.05, end=0.95, step=0.05, value=0.50, sizing_mode="stretch_width")

mod_allena = pn.widgets.Button(name="▶  Allena e valuta i modelli", button_type="primary", height=44, sizing_mode="stretch_width")
mod_esito = pn.pane.Alert("", visible=False, sizing_mode="stretch_width")
mod_dettaglio = pn.widgets.Select(name="Modello da analizzare nel dettaglio", options=[], sizing_mode="stretch_width")
mod_dettaglio_area = pn.Column(sizing_mode="stretch_width")
area_modelli = pn.Column(sizing_mode="stretch_width")



def imposta_feature_default(event=None):
    target = mod_target.value
    mod_feature.value = [c for c in mod_feature.options if c != target]


def aggiorna_modelli():
    risultati_modelli["ultimo"] = None
    mod_esito.visible = False
    df = stato["pulito"]

    if not SKLEARN_DISPONIBILE:
        mod_allena.disabled = True
        area_modelli.objects = [
            pn.pane.Alert("scikit-learn non è installato (`pip install scikit-learn catboost`).", alert_type="danger")
        ]
        return

    area_modelli.objects = [
        pn.pane.Markdown("_Configura il target e i modelli, quindi premi «Allena e valuta i modelli»._")
    ]
    mod_dettaglio_area.objects = []

    if df is None:
        return

    colonne = list(df.columns)
    maschera = maschera_mancanti(df)

    candidati, binari = [], []
    for colonna in colonne:
        distinti = df.loc[~maschera[colonna], colonna].nunique()
        if 2 <= distinti <= MAX_CLASSI:
            candidati.append(colonna)
            if distinti == 2:
                binari.append(colonna)

    mod_feature.value = []
    mod_feature.options = colonne
    mod_target.options = candidati

    if len(candidati) > 0:
        if "Depression" in candidati:
            mod_target.value = "Depression"
        else:
            mod_target.value = binari[0] if len(binari) > 0 else candidati[0]
    else:
        mod_target.value = None
        mostra_esito(mod_esito, f"Nessuna colonna adatta come target (2-{MAX_CLASSI} valori distinti).", "warning")

    imposta_feature_default()
    mod_allena.disabled = len(candidati) == 0


def valore_o(widget, predefinito):
    return predefinito if widget.value is None else widget.value


def leggi_configurazione():
    return {
        "target": mod_target.value,
        "feature": list(mod_feature.value),
        "test_size": float(mod_test_size.value),
        "soglia": float(mod_soglia.value),
        "ottimizza_soglia": bool(mod_ottimizza_soglia.value),
        "seed": int(valore_o(mod_seed, 42)),
        "stratifica": bool(mod_stratifica.value),
        "cv": bool(mod_cv.value),
        "modelli": [
            nome for nome, attivo in [
                ("Regressione logistica", mod_usa_lr.value),
                ("Albero decisionale", mod_usa_dt.value),
                ("Random forest", mod_usa_rf.value),
                ("CatBoost", mod_usa_cb.value),
            ] if attivo
        ],
        "parametri": {
            "use_class_weights": bool(mod_class_weight.value),
            "lr_c": float(par_lr_c.value),
            "dt_criterion": par_dt_criterion.value,
            "dt_profondita": int(par_dt_depth.value) or None,
            "dt_min_samples_split": int(par_dt_split.value),
            "dt_min_samples_leaf": int(par_dt_leaf.value),
            "dt_ccp_alpha": float(par_dt_alpha.value),
            "rf_alberi": int(par_rf_trees.value),
            "rf_profondita": int(par_rf_depth.value) or None,
            "rf_min_samples_split": int(par_rf_split.value),
            "rf_min_samples_leaf": int(par_rf_leaf.value),
            "rf_max_features": None if par_rf_features.value == "None" else par_rf_features.value,
            "cb_iterazioni": int(par_cb_iterations.value),
            "cb_profondita": int(par_cb_depth.value),
            "cb_learning_rate": float(par_cb_lr.value),
            "cb_l2": float(par_cb_l2.value),
        }
    }


async def allena_modelli(event=None):
    df = stato["pulito"]
    if df is None or not SKLEARN_DISPONIBILE:
        return

    config = leggi_configurazione()
    if config["target"] is None:
        mostra_esito(mod_esito, "Seleziona una variabile target.", "warning")
        return

    if len(config["modelli"]) == 0:
        mostra_esito(mod_esito, "Seleziona almeno un modello.", "warning")
        return

    mod_esito.object = "⏳ Addestramento in corso… i modelli selezionati stanno imparando dai dati."
    mod_esito.alert_type = "info"
    mod_esito.visible = True
    mod_allena.disabled = True
    area_modelli.loading = True

    await asyncio.sleep(0.05)
    risultati = None

    try:
        risultati = await asyncio.to_thread(addestra_modelli, df.copy(), config)
    except Exception as errore:
        traceback.print_exc()
        mostra_esito(mod_esito, f"**Errore durante l'addestramento:** {errore}", "danger")
    finally:
        area_modelli.loading = False
        mod_allena.disabled = False

    if risultati is not None:
        mostra_risultati_modelli(risultati)
        aggiorna_dashboard_home()


def mostra_risultati_modelli(ris):
    risultati_modelli["ultimo"] = ris
    mod_esito.object = (
        f"Addestramento completato: {len(ris['modelli'])} modelli disponibili "
        f"su {len(MODELLI)} richiesti."
    )
    mod_esito.alert_type = "success" if not ris["errori"] else "warning"
    mod_esito.visible = True
    classi = ris["classi"]
    tabella = ris["tabella"]

    distribuzione = ", ".join(f"{c}: {n}" for c, n in ris["conteggio_classi"].items())
    riepilogo = (
        f"**Target:** `{ris['target']}` ({len(classi)} classi — {distribuzione})  \n"
        f"**Split:** {ris['numero_train']} righe train / {ris['numero_test']} righe test  \n"
        f"**Variabili usate ({len(ris['feature_usate'])}):** {', '.join(ris['feature_usate'])}"
    )

    if len(ris["escluse"]) > 0:
        riepilogo += f"  \n**Variabili escluse:** {', '.join(ris['escluse'])}"

    oggetti = [pn.pane.Markdown(riepilogo)]

    if len(ris["avvisi"]) > 0:
        oggetti.append(pn.pane.Alert("\n".join(f"- {a}" for a in ris["avvisi"]), alert_type="warning"))

    if len(ris["errori"]) > 0:
        oggetti.append(pn.pane.Alert("Modelli non addestrati:\n" + "\n".join(f"- **{n}**: {t}" for n, t in ris["errori"].items()), alert_type="danger"))

    oggetti.append(pn.pane.Markdown("### Confronto tra i modelli"))
    oggetti.append(pn.widgets.Tabulator(tabella, show_index=False, disabled=True, height=70 + 36 * len(tabella), sizing_mode="stretch_width"))

    metriche = tabella.melt(
        id_vars="Modello", value_vars=["Accuracy test", "Precision", "Recall", "F1", "ROC AUC"],
        var_name="Metrica", value_name="Valore"
    ).dropna()

    fig_metriche = px.bar(metriche, x="Modello", y="Valore", color="Metrica", barmode="group", title="Metriche sul test set")
    fig_metriche.update_yaxes(range=[0, 1.05])
    impostazioni_comuni(fig_metriche)

    grafici = [pn.pane.Plotly(fig_metriche, height=380, sizing_mode="stretch_width", config={"responsive": True})]

    if len(classi) == 2:
        fig_roc = go.Figure()
        for nome, dettaglio in ris["modelli"].items():
            if dettaglio["roc"] is None:
                continue
            fpr, tpr, auc = dettaglio["roc"]
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nome} (AUC {auc:.3f})"))

        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Caso (AUC 0.5)", line=dict(dash="dash", color="#999999")))
        fig_roc.update_layout(title=f"Curve ROC (classe positiva: {classi[1]})", xaxis_title="False positive rate", yaxis_title="True positive rate")
        impostazioni_comuni(fig_roc)

        grafici.append(pn.pane.Plotly(fig_roc, height=380, sizing_mode="stretch_width", config={"responsive": True}))

    oggetti.append(pn.Row(*grafici, sizing_mode="stretch_width"))
    oggetti.append(pn.layout.Divider())
    area_modelli.objects = oggetti
    nomi = list(ris["modelli"].keys())
    mod_dettaglio.options = nomi
    mod_dettaglio.value = nomi[0]

    mostra_dettaglio_modello()


def mostra_dettaglio_modello(event=None):
    ris = risultati_modelli["ultimo"]
    nome = mod_dettaglio.value

    if ris is None or nome not in ris["modelli"]:
        mod_dettaglio_area.objects = []
        return

    dettaglio = ris["modelli"][nome]
    classi = ris["classi"]
    riga = ris["tabella"].set_index("Modello").loc[nome]

    fig_matrice = px.imshow(
        dettaglio["matrice"], x=classi, y=classi, text_auto=True,
        color_continuous_scale="Blues", aspect="auto",
        labels=dict(x="Predetto", y="Reale", color="Conteggio"),
        title=f"Matrice di confusione (test set) – {nome}"
    )
    impostazioni_comuni(fig_matrice)

    importanze = dettaglio["importanze"].sort_values(ascending=False).head(15).iloc[::-1]
    fig_importanze = px.bar(
        x=importanze.values, y=importanze.index, orientation="h",
        labels={"x": "Importanza", "y": ""}, title=f"{dettaglio['titolo_importanze']} – {nome}"
    )
    impostazioni_comuni(fig_importanze)

    soglia_str = f" · soglia {riga['Soglia']:.2f}" if "Soglia" in riga else ""
    accuracy_train = "n/d" if pd.isna(riga["Accuracy train"]) else f"{riga['Accuracy train']:.3f}"
    oggetti = [
        pn.pane.Markdown(
            f"Accuracy train **{accuracy_train}** · accuracy test **{riga['Accuracy test']:.3f}** · "
            f"F1 **{riga['F1']:.3f}**{soglia_str} · tempo **{riga['Tempo (s)']:.2f} s**"
        ),
        pn.Row(
            pn.pane.Plotly(fig_matrice, height=380, sizing_mode="stretch_width", config={"responsive": True}),
            pn.pane.Plotly(fig_importanze, height=380, sizing_mode="stretch_width", config={"responsive": True}),
            sizing_mode="stretch_width"
        ),
        pn.pane.Markdown("#### Report per classe (test set)"),
        pn.widgets.Tabulator(dettaglio["report"], show_index=False, disabled=True, height=min(400, 70 + 36 * len(dettaglio["report"])), sizing_mode="stretch_width")
    ]

    if dettaglio["testo_albero"] is not None:
        legenda = ", ".join(f"{idx} = {cl}" for idx, cl in enumerate(classi))
        oggetti.append(
            pn.pane.Markdown(
                f"#### Regole dell'albero (primi 4 livelli)\nCodici classi: {legenda}\n\n```\n{dettaglio['testo_albero']}\n```"
            )
        )

    mod_dettaglio_area.objects = oggetti


mod_target.param.watch(imposta_feature_default, "value")
mod_dettaglio.param.watch(mostra_dettaglio_modello, "value")
mod_allena.on_click(allena_modelli)


# ============================================================
# SCHEDA "MODELLI": LAYOUT
# ============================================================

avviso_librerie = []
if not CATBOOST_DISPONIBILE:
    avviso_librerie.append(
        pn.pane.Alert(
            "CatBoost non è installato (`pip install catboost`).",
            alert_type="warning", sizing_mode="stretch_width"
        )
    )

scheda_modelli = pn.Column(
    pn.pane.Markdown(
        "### Modelli di classificazione\n"
        "I quattro modelli vengono addestrati automaticamente dopo il caricamento del CSV, "
        "utilizzando lo split, il preprocessing e i migliori parametri definiti in `main.py` "
        "(e salvati nei file `.par`). `Health_condition` è esclusa da tutte le feature predittive "
        "per evitare target leakage."
    ),
    mod_esito,
    area_modelli,
    pn.layout.Divider(),
    pn.pane.Markdown("### Dettaglio del modello"),
    mod_dettaglio,
    mod_dettaglio_area,
    styles={"background": "#f8fafc", "padding": "18px", "border-radius": "14px"},
    sizing_mode="stretch_width"
)


# ============================================================
# SCHEDA "PULIZIA DATI": LAYOUT
# ============================================================

tab_colonne = pn.Column(
    pn.pane.Markdown("### Colonne\nSeleziona le colonne non necessarie."),
    col_informazioni,
    pn.layout.Divider(),
    pn.Row(
        pn.Column(col_selezione, col_rimuovi, col_conferma, width=340),
        pn.Column(col_esito, sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA, sizing_mode="stretch_width"
)

tab_duplicati = pn.Column(
    pn.pane.Markdown("### Righe duplicate\nScegli le colonne da confrontare."),
    pn.Row(
        pn.Column(dup_colonne, dup_mantieni, pn.Row(dup_cerca, dup_rimuovi), dup_conferma, width=340),
        pn.Column(dup_esito, dup_risultato, sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA, sizing_mode="stretch_width"
)

tab_mancanti = pn.Column(
    pn.pane.Markdown("### Valori mancanti"),
    mis_statistiche,
    pn.layout.Divider(),
    pn.Row(
        pn.Column(mis_colonne, pn.Row(mis_mostra, mis_rimuovi), pn.layout.Divider(), mis_metodo, mis_valore, mis_imputa, mis_conferma, width=340),
        pn.Column(mis_esito, mis_risultato, sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA, sizing_mode="stretch_width"
)

tab_outlier = pn.Column(
    pn.pane.Markdown("### Outlier\nIndividua e rimuovi gli outlier."),
    pn.Row(
        pn.Column(out_colonna, out_metodo, pn.Row(out_cerca, out_rimuovi), out_conferma, width=340),
        pn.Column(out_esito, out_risultato, sizing_mode="stretch_width"),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA, sizing_mode="stretch_width"
)

tab_dataset = pn.Column(
    pn.pane.Markdown("### Dataset pulito\nRiepilogo delle operazioni ed esportazione CSV."),
    riepilogo_dataset,
    pn.pane.Markdown("#### Operazioni eseguite"),
    registro_operazioni,
    pn.layout.Divider(),
    pn.Row(selettore_separatore, pn.Column(pn.Spacer(height=20), scarica), ripristina),
    ripristino_conferma,
    ripristino_esito,
    styles=STILE_SCHEDA, sizing_mode="stretch_width"
)

sottoschede_pulizia = pn.Tabs(
    ("Colonne", tab_colonne),
    ("Duplicati", tab_duplicati),
    ("Valori mancanti", tab_mancanti),
    ("Outlier", tab_outlier),
    ("Dataset pulito", tab_dataset),
    dynamic=True, sizing_mode="stretch_width"
)

pulizia = pn.Column(barra_stato, sottoschede_pulizia, sizing_mode="stretch_width")

schede_student = pn.Tabs(
    ("Dashboard", dashboard_home),
    ("Esplorazione", esplorazione),
    ("Pulizia dati", pulizia),
    ("Modelli", scheda_modelli),
    ("Report", report_main),
    dynamic=True, sizing_mode="stretch_width"
)

# Per dataset diversi da Student Depression l'addestramento è manuale:
# mostriamo esplicitamente i controlli di configurazione e il pulsante.
selezione_modelli = pn.Row(
    pn.Column(mod_usa_lr, styles={"border": "1px solid #d9e2f0", "padding": "8px", "border-radius": "8px"}),
    pn.Column(mod_usa_dt, styles={"border": "1px solid #d9e2f0", "padding": "8px", "border-radius": "8px"}),
    pn.Column(mod_usa_rf, styles={"border": "1px solid #d9e2f0", "padding": "8px", "border-radius": "8px"}),
    pn.Column(mod_usa_cb, styles={"border": "1px solid #d9e2f0", "padding": "8px", "border-radius": "8px"}),
    sizing_mode="stretch_width"
)

pannello_parametri_modelli = pn.Accordion(
    ("Impostazioni regressione logistica", pn.Column(pn.Row(par_lr_c))),
    ("Impostazioni albero decisionale", pn.Column(pn.Row(par_dt_criterion, par_dt_depth), pn.Row(par_dt_split, par_dt_leaf, par_dt_alpha))),
    ("Impostazioni random forest", pn.Column(pn.Row(par_rf_trees, par_rf_depth), pn.Row(par_rf_split, par_rf_leaf, par_rf_features))),
    ("Impostazioni CatBoost", pn.Column(pn.Row(par_cb_iterations, par_cb_depth), pn.Row(par_cb_lr, par_cb_l2))),
    active=[], sizing_mode="stretch_width"
)

scheda_modelli_generica = pn.Column(
    pn.pane.Markdown(
        "### Addestramento manuale dei modelli\n"
        "Seleziona target e feature, abilita uno o più algoritmi e modifica i rispettivi iperparametri. "
        "Quando premi il pulsante, il training parte immediatamente e i risultati vengono mostrati qui sotto."
    ),
    pn.Row(mod_target, mod_feature, sizing_mode="stretch_width"),
    pn.pane.Markdown("#### Seleziona i modelli da addestrare"),
    selezione_modelli,
    pn.pane.Markdown("#### Impostazioni dei singoli modelli (apri per modificare gli iperparametri)"),
    pannello_parametri_modelli,
    pn.pane.Markdown("#### Configurazione dello split e della valutazione"),
    pn.Row(mod_test_size, mod_seed, mod_soglia, sizing_mode="stretch_width"),
    pn.Row(mod_stratifica, mod_cv, mod_class_weight, mod_ottimizza_soglia, sizing_mode="stretch_width"),
    mod_allena,
    mod_esito,
    area_modelli,
    pn.layout.Divider(),
    pn.pane.Markdown("### Dettaglio del modello"),
    mod_dettaglio,
    mod_dettaglio_area,
    styles={"background": "#f8fafc", "padding": "18px", "border-radius": "14px"},
    sizing_mode="stretch_width"
)

schede_generiche = pn.Tabs(
    ("Esplorazione", esplorazione),
    ("Pulizia dati", pulizia),
    ("Modelli", scheda_modelli_generica),
    dynamic=True, sizing_mode="stretch_width"
)


# ============================================================
# CARICAMENTO E AGGIORNAMENTO DEL DATASET
# ============================================================

def avvia_addestramento_automatico():
    """Avvia i modelli Student Depression appena il dataset è disponibile."""
    if stato["pulito"] is None:
        return

    if "Depression" not in stato["pulito"].columns:
        mostra_esito(
            mod_esito,
            "Il dataset è stato caricato, ma non contiene la colonna `Depression`: "
            "l'addestramento automatico dei modelli Student Depression non è applicabile.",
            "warning"
        )
        return

    mod_target.value = "Depression"
    mod_scelta.value = list(MODELLI)
    mod_usa_lr.value = True
    mod_usa_dt.value = True
    mod_usa_rf.value = True
    mod_usa_cb.value = CATBOOST_DISPONIBILE
    mod_esito.object = "Addestramento automatico dei quattro modelli con i parametri di `main.py`..."
    mod_esito.alert_type = "info"
    mod_esito.visible = True
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(allena_modelli())
    except RuntimeError:
        pn.state.execute(lambda: asyncio.create_task(allena_modelli()))


def carica_dataset(event):
    if file_input.value is None:
        return

    try:
        df = leggi_csv(file_input.value)
        if df.empty:
            messaggio.object = "## Attenzione\n\nIl CSV è stato letto, ma non contiene righe di dati."
            main.objects = [messaggio]
            return

        stato["originale"] = df
        stato["pulito"] = df.copy()
        stato["registro"] = []
        risultati_modelli["ultimo"] = None

        nome_file = file_input.filename or "dataset.csv"
        stato["nome_file"] = nome_file
        scarica.filename = os.path.splitext(nome_file)[0] + "_pulito.csv"

        aggiorna_tutto()

        if nome_file == "Student_Depression_Dataset.csv":
            main.objects = [nome_dataset, schede_student]
            avvia_addestramento_automatico()
        else:
            risultati_modelli["ultimo"] = None
            mostra_esito(
                mod_esito,
                "**Dataset diverso da Student_Depression_Dataset.csv.** Dashboard iniziale e report predefiniti sono disattivati. "
                "Puoi comunque esplorare, pulire il CSV e addestrare manualmente i modelli scegliendo target e feature.",
                "warning"
            )
            main.objects = [nome_dataset, pn.pane.Alert(
                "Dashboard e report Student Depression non disponibili per questo file.",
                alert_type="warning", sizing_mode="stretch_width"
            ), schede_generiche]

        print(f"Dataset caricato: {df.shape[0]} righe, {df.shape[1]} colonne.")

    except Exception as errore:
        traceback.print_exc()
        messaggio_errore = pn.pane.Markdown(
            f"## Errore durante il caricamento\n\n"
            f"**Tipo di errore:** `{type(errore).__name__}`\n\n"
            f"**Dettaglio:** `{str(errore)}`\n"
        )
        main.objects = [messaggio_errore]


# ============================================================
# SIDEBAR E TEMPLATE
# ============================================================

sidebar = pn.Column(
    pn.pane.Markdown("# Data Explorer"),
    pn.pane.Markdown("Carica un CSV per visualizzare, analizzare e pulire i dati."),
    file_input,
    pn.pane.Markdown(
        """
        ---
        ### Funzionalità

        - Dashboard e report automatici per Student Depression
        - Panoramica e statistiche del dataset
        - Grafici interattivi personalizzabili
        - Pulizia dati (mancanti, duplicati, outlier)
        - Modelli ML sincronizzati con `main.py`
        """
    ),
    sizing_mode="stretch_width"
)

dashboard = pn.template.FastListTemplate(
    title=TITOLO,
    sidebar=[sidebar],
    main=[main],
    accent_base_color="#2563eb",
    header_background="#1f2937"
)

file_input.param.watch(carica_dataset, "value")
dashboard.servable()
