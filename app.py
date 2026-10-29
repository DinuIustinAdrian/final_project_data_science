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

# Librerie per la scheda "Modelli". Se non sono installate l'app parte
# comunque e la scheda spiega cosa installare:
#     pip install scikit-learn catboost
try:
    from sklearn.base import clone
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
        accuracy_score
    )
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (
        LabelEncoder,
        OneHotEncoder,
        StandardScaler
    )
    from sklearn.tree import DecisionTreeClassifier, export_text

    SKLEARN_DISPONIBILE = True

except ImportError:
    SKLEARN_DISPONIBILE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_DISPONIBILE = True

except ImportError:
    CATBOOST_DISPONIBILE = False

# MODIFICA: aggiunta l'estensione "tabulator", necessaria per usare
# pn.widgets.Tabulator in modo affidabile.
pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


# ============================================================
# CONFIGURAZIONE
# ============================================================

TITOLO = "Data Explorer"
NUMERO_GRAFICI = 6
MAX_CATEGORIE = 20

# MODIFICA: il limite era fisso a 2000 righe dentro crea_grafico().
# Con un campione cosi piccolo le frequenze assolute sarebbero false
# per i dataset grandi, quindi il limite e ora una costante piu alta.
MAX_RIGHE_GRAFICO = 50000

# Numero massimo di righe mostrate nelle tabelle della pulizia dati.
MAX_RIGHE_TABELLA = 2000

# Numero massimo di punti "normali" disegnati nel grafico degli outlier
# (gli outlier vengono sempre disegnati tutti).
MAX_PUNTI_OUTLIER = 5000

# Scheda "Modelli": numero massimo di classi del target, numero massimo
# di valori distinti di una variabile categorica usata come feature e
# modelli disponibili.
MAX_CLASSI = 20
MAX_CATEGORIE_FEATURE = 50

MODELLI = [
    "Regressione logistica",
    "Albero decisionale",
    "Random forest",
    "CatBoost",
    "K-nearest neighbors"
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
# "originale": dataset appena caricato, non viene mai modificato.
# "pulito":    copia di lavoro su cui agiscono tutte le operazioni.
# "registro":  elenco testuale delle operazioni eseguite.

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

    separatori = [None, ",", ";", "\t", "|"]
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

                # Evita di accettare file che risultano completamente vuoti.
                if len(df.columns) == 0:
                    continue

                # Rimuove eventuali spazi nei nomi delle colonne.
                df.columns = [
                    str(colonna).strip()
                    for colonna in df.columns
                ]

                # Rimuove le colonne senza nome create da alcuni CSV.
                colonne_valide = [
                    colonna
                    for colonna in df.columns
                    if colonna and not colonna.startswith("Unnamed:")
                ]

                df = df.loc[:, colonne_valide]

                if len(df.columns) == 0:
                    continue

                # Se il separatore non è stato riconosciuto,
                # prova il successivo.
                if len(df.columns) == 1:
                    valore = str(df.columns[0])

                    if any(
                        simbolo in valore
                        for simbolo in [",", ";", "\t", "|"]
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
    è mancante. Le stringhe vuote o fatte solo di spazi sono considerate
    mancanti (come già avveniva nei grafici).
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
    Restituisce: il riquadro, la funzione per mostrarlo e quella per nasconderlo.
    La funzione passata a chiedi() viene eseguita solo dopo la conferma.
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
# STATISTICHE DEL DATASET
# ============================================================

def crea_statistiche(df):

    numero_righe = len(df)
    numero_colonne = len(df.columns)

    # MODIFICA: usa maschera_mancanti() cosi i numeri sono gli stessi
    # della scheda di pulizia (stringhe vuote incluse).
    maschera = maschera_mancanti(df)

    celle_mancanti = int(maschera.sum().sum())

    totale_celle = numero_righe * numero_colonne

    if totale_celle > 0:
        percentuale_mancanti = (
            celle_mancanti / totale_celle
        ) * 100
    else:
        percentuale_mancanti = 0

    righe_incomplete = int(maschera.any(axis=1).sum())
    duplicati = int(df.duplicated().sum())

    colonne_numeriche = len(
        df.select_dtypes(include="number").columns
    )

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
                <div style="
                    font-size: 13px;
                    color: #666666;
                    margin-bottom: 8px;
                ">
                    {titolo}
                </div>

                <div style="
                    font-size: 24px;
                    font-weight: bold;
                    color: #222222;
                ">
                    {valore}
                </div>
            </div>
            """,
            sizing_mode="stretch_width"
        )

        schede.append(scheda)

    return pn.GridBox(
        *schede,
        ncols=4,
        sizing_mode="stretch_width"
    )


# ============================================================
# STATISTICHE DESCRITTIVE
# ============================================================

def crea_tabella_statistiche(df):

    if df.empty or len(df.columns) == 0:
        return pn.pane.Markdown("Nessun dato disponibile.")

    statistiche = df.describe(
        include="all"
    ).transpose()

    statistiche = statistiche.reset_index()

    statistiche = statistiche.rename(
        columns={"index": "Variabile"}
    )

    tabella = pn.widgets.Tabulator(
        statistiche,
        show_index=False,
        pagination="local",
        page_size=15,
        height=350,
        sizing_mode="stretch_width"
    )

    return tabella


# ============================================================
# CREAZIONE DEL GRAFICO
# ============================================================

def crea_grafico(df, colonna, gruppo=None):

    if colonna not in df.columns:

        fig = px.scatter(
            title="Seleziona una variabile valida"
        )

        return fig

    dati = df.copy()

    # Limito il numero di righe utilizzate per il grafico.
    # MODIFICA: se il dataset viene campionato lo scrivo nel titolo,
    # perche le frequenze mostrate sarebbero quelle del campione.
    nota_campione = ""

    if len(dati) > MAX_RIGHE_GRAFICO:
        dati = dati.sample(
            n=MAX_RIGHE_GRAFICO,
            random_state=42
        )

        nota_campione = f" (campione di {MAX_RIGHE_GRAFICO} righe)"


    # Rimuove gli spazi dalle stringhe.
    # Non modifica il DataFrame originale.
    for nome_colonna in dati.columns:

        if (
            pd.api.types.is_object_dtype(dati[nome_colonna])
            or isinstance(
                dati[nome_colonna].dtype,
                pd.StringDtype
            )
        ):

            dati[nome_colonna] = dati[nome_colonna].replace(
                r"^\s*$",
                np.nan,
                regex=True
            )

    # Elimina le righe mancanti solo per la variabile analizzata.
    dati = dati.dropna(subset=[colonna])

    if dati.empty:

        fig = px.scatter(
            title=f"Nessun dato disponibile per {colonna}"
        )

        return fig

    # Verifica se la variabile è numerica.
    variabile_numerica = pd.api.types.is_numeric_dtype(
        dati[colonna]
    )

    # Se una variabile contiene numeri memorizzati come stringhe,
    # prova a convertirla per il grafico.
    if not variabile_numerica:

        conversione = pd.to_numeric(
            dati[colonna],
            errors="coerce"
        )

        valori_originali = dati[colonna].notna().sum()
        valori_convertiti = conversione.notna().sum()

        if (
            valori_originali > 0
            and valori_convertiti / valori_originali >= 0.9
        ):

            dati[colonna] = conversione
            dati = dati.dropna(subset=[colonna])

            variabile_numerica = True

    # La variabile di raggruppamento deve essere diversa
    # dalla variabile principale.
    if gruppo == colonna:
        gruppo = None

    if gruppo not in dati.columns:
        gruppo = None

    # --------------------------------------------------------
    # GRAFICO NUMERICO
    # --------------------------------------------------------

    if variabile_numerica:

        if gruppo is not None:

            dati = dati.dropna(subset=[gruppo])

            if dati.empty:

                return px.scatter(
                    title="Nessun dato disponibile per il confronto"
                )

            # MODIFICA: il gruppo è trattato come testo e limitato alle
            # categorie piu frequenti (evita centinaia di serie se il
            # gruppo è una variabile continua).
            dati[gruppo] = dati[gruppo].astype(str)

            gruppi_principali = (
                dati[gruppo]
                .value_counts()
                .head(MAX_CATEGORIE)
                .index
            )

            dati = dati[
                dati[gruppo].isin(gruppi_principali)
            ].copy()

            # MODIFICA: niente piu histnorm="percent": frequenze assolute.
            fig = px.histogram(
                dati,
                x=colonna,
                color=gruppo,
                barmode="group",
                opacity=0.65,
                nbins=30,
                title=(
                    f"Distribuzione di {colonna} per {gruppo}"
                    f"{nota_campione}"
                ),
                labels={
                    colonna: colonna,
                    gruppo: gruppo
                }
            )

            fig.update_layout(
                yaxis_title="Frequenza",
                legend_title_text=gruppo
            )

        else:

            fig = px.histogram(
                dati,
                x=colonna,
                nbins=30,
                title=f"Distribuzione di {colonna}{nota_campione}"
            )

            fig.update_layout(
                yaxis_title="Frequenza"
            )

    # --------------------------------------------------------
    # GRAFICO CATEGORICO
    # --------------------------------------------------------

    else:

        dati[colonna] = dati[colonna].astype(str)

        # Limita il numero di categorie visualizzate.
        categorie_principali = (
            dati[colonna]
            .value_counts()
            .head(MAX_CATEGORIE)
            .index
        )

        dati = dati[
            dati[colonna].isin(categorie_principali)
        ].copy()

        if gruppo is not None:

            dati[gruppo] = dati[gruppo].fillna(
                "(Mancante)"
            ).astype(str)

            # MODIFICA: anche qui il gruppo è limitato alle categorie
            # piu frequenti.
            gruppi_principali = (
                dati[gruppo]
                .value_counts()
                .head(MAX_CATEGORIE)
                .index
            )

            dati = dati[
                dati[gruppo].isin(gruppi_principali)
            ].copy()

            conteggi = pd.crosstab(
                dati[colonna],
                dati[gruppo]
            )

            fig = px.bar(
                conteggi,
                barmode="group",
                title=f"{colonna} per {gruppo}{nota_campione}",
                labels={
                    "index": colonna,
                    colonna: colonna,
                    "value": "Frequenza"
                }
            )

            fig.update_layout(
                xaxis_title=colonna,
                yaxis_title="Frequenza",
                legend_title_text=gruppo
            )

        else:

            conteggi = (
                dati[colonna]
                .value_counts()
                .reset_index()
            )

            conteggi.columns = [
                colonna,
                "Frequenza"
            ]

            fig = px.bar(
                conteggi,
                x=colonna,
                y="Frequenza",
                title=f"Frequenza di {colonna}{nota_campione}"
            )

            fig.update_layout(
                xaxis_title=colonna,
                yaxis_title="Frequenza"
            )

    # Impostazioni comuni.
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(
            l=40,
            r=20,
            t=60,
            b=60
        ),
        font=dict(size=11)
    )

    fig.update_xaxes(
        automargin=True
    )

    fig.update_yaxes(
        automargin=True
    )

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

# Per ogni tipo: serve la variabile Y? Si puo colorare per gruppo?
# Serve la funzione di aggregazione?
TIPI_GRAFICO = {
    TIPO_AUTOMATICO: {"y": False, "gruppo": True, "aggregazione": False},
    TIPO_SCATTER: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_LINEE: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_BOX: {"y": True, "gruppo": True, "aggregazione": False},
    TIPO_BARRE: {"y": True, "gruppo": True, "aggregazione": True},
    TIPO_HEATMAP: {"y": True, "gruppo": False, "aggregazione": False},
}

FUNZIONI_AGGREGAZIONE = {
    "Media": "mean",
    "Mediana": "median",
    "Somma": "sum"
}

PALETTE_COLORI = px.colors.qualitative.Plotly


def figura_vuota(testo):
    return px.scatter(title=testo)


def converti_in_numerica(dati, colonna):
    """
    True se la colonna è numerica oppure convertibile in numeri
    (almeno il 90% dei valori, stessa regola di crea_grafico).
    In questo caso la colonna di "dati" viene convertita.
    """

    if pd.api.types.is_numeric_dtype(dati[colonna]):
        return True

    conversione = pd.to_numeric(dati[colonna], errors="coerce")

    valori_originali = dati[colonna].notna().sum()
    valori_convertiti = conversione.notna().sum()

    if (
        valori_originali > 0
        and valori_convertiti / valori_originali >= 0.9
    ):
        dati[colonna] = conversione
        return True

    return False


def limita_categorie(dati, colonna):
    """Trasforma la colonna in testo e tiene le categorie piu frequenti."""

    dati = dati.copy()

    dati[colonna] = dati[colonna].astype(str)

    principali = (
        dati[colonna]
        .value_counts()
        .head(MAX_CATEGORIE)
        .index
    )

    return dati[dati[colonna].isin(principali)].copy()


def impostazioni_comuni(fig):

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(
            l=40,
            r=20,
            t=60,
            b=60
        ),
        font=dict(size=11)
    )

    fig.update_xaxes(
        automargin=True
    )

    fig.update_yaxes(
        automargin=True
    )

    return fig


def crea_grafico_due_variabili(
    df,
    tipo,
    colonna_x,
    colonna_y,
    gruppo=None,
    aggregazione="Media"
):

    if (
        colonna_x not in df.columns
        or colonna_y not in df.columns
        or colonna_x == colonna_y
    ):
        return figura_vuota("Seleziona due variabili distinte")

    if (
        tipo == TIPO_HEATMAP
        or gruppo in (colonna_x, colonna_y)
        or gruppo not in df.columns
    ):
        gruppo = None

    colonne_usate = [colonna_x, colonna_y]

    if gruppo is not None:
        colonne_usate.append(gruppo)

    dati = df[colonne_usate].copy()

    nota_campione = ""

    if len(dati) > MAX_RIGHE_GRAFICO:
        dati = dati.sample(
            n=MAX_RIGHE_GRAFICO,
            random_state=42
        )

        nota_campione = f" (campione di {MAX_RIGHE_GRAFICO} righe)"

    # Stringhe vuote = valori mancanti.
    for nome_colonna in dati.columns:

        if (
            pd.api.types.is_object_dtype(dati[nome_colonna])
            or isinstance(
                dati[nome_colonna].dtype,
                pd.StringDtype
            )
        ):

            dati[nome_colonna] = dati[nome_colonna].replace(
                r"^\s*$",
                np.nan,
                regex=True
            )

    # La variabile Y deve sempre essere numerica.
    if not converti_in_numerica(dati, colonna_y):
        return figura_vuota(
            f"{tipo}: la variabile Y «{colonna_y}» deve essere numerica"
        )

    if tipo in (TIPO_SCATTER, TIPO_HEATMAP):

        if not converti_in_numerica(dati, colonna_x):
            return figura_vuota(
                f"{tipo}: la variabile X «{colonna_x}» deve essere numerica"
            )

    if tipo == TIPO_LINEE:

        # X puo essere un numero oppure una data.
        if not converti_in_numerica(dati, colonna_x):

            date = pd.to_datetime(
                dati[colonna_x],
                errors="coerce"
            )

            valide = dati[colonna_x].notna().sum()

            if valide > 0 and date.notna().sum() >= 0.9 * valide:
                dati[colonna_x] = date
            else:
                return figura_vuota(
                    f"{tipo}: la variabile X «{colonna_x}» "
                    f"deve essere numerica o una data"
                )

    dati = dati.dropna(subset=colonne_usate)

    if dati.empty:
        return figura_vuota("Nessun dato disponibile per questa combinazione")

    if gruppo is not None:
        dati = limita_categorie(dati, gruppo)

    per_gruppo = f" per {gruppo}" if gruppo is not None else ""

    # --------------------------------------------------------
    if tipo == TIPO_SCATTER:

        fig = px.scatter(
            dati,
            x=colonna_x,
            y=colonna_y,
            color=gruppo,
            opacity=0.7,
            title=(
                f"{colonna_y} in funzione di {colonna_x}"
                f"{per_gruppo}{nota_campione}"
            )
        )

    # --------------------------------------------------------
    elif tipo == TIPO_LINEE:

        dati = dati.sort_values(colonna_x)

        fig = px.line(
            dati,
            x=colonna_x,
            y=colonna_y,
            color=gruppo,
            title=(
                f"{colonna_y} in funzione di {colonna_x}"
                f"{per_gruppo}{nota_campione}"
            )
        )

    # --------------------------------------------------------
    elif tipo == TIPO_BOX:

        dati = limita_categorie(dati, colonna_x)

        fig = px.box(
            dati,
            x=colonna_x,
            y=colonna_y,
            color=gruppo,
            title=(
                f"Distribuzione di {colonna_y} per {colonna_x}"
                f"{per_gruppo}{nota_campione}"
            )
        )

    # --------------------------------------------------------
    elif tipo == TIPO_BARRE:

        dati = limita_categorie(dati, colonna_x)

        funzione = FUNZIONI_AGGREGAZIONE.get(aggregazione, "mean")

        colonne_raggruppamento = [colonna_x]

        if gruppo is not None:
            colonne_raggruppamento.append(gruppo)

        sintesi = (
            dati
            .groupby(colonne_raggruppamento)[colonna_y]
            .agg(funzione)
            .reset_index()
        )

        etichetta = f"{aggregazione} di {colonna_y}"

        sintesi = sintesi.rename(
            columns={colonna_y: etichetta}
        )

        fig = px.bar(
            sintesi,
            x=colonna_x,
            y=etichetta,
            color=gruppo,
            barmode="group",
            title=(
                f"{etichetta} per {colonna_x}"
                f"{per_gruppo}{nota_campione}"
            )
        )

    # --------------------------------------------------------
    else:

        # Heatmap di densita: frequenze assolute.
        fig = px.density_heatmap(
            dati,
            x=colonna_x,
            y=colonna_y,
            nbinsx=30,
            nbinsy=30,
            color_continuous_scale=["#ffffff", "#2563eb"],
            title=(
                f"Densità di {colonna_y} e {colonna_x}{nota_campione}"
            )
        )

        fig.update_layout(
            coloraxis_colorbar_title_text="Frequenza"
        )

    return impostazioni_comuni(fig)


def crea_grafico_scelto(
    df,
    tipo,
    colonna_x,
    colonna_y,
    gruppo,
    aggregazione
):
    """Sceglie la funzione giusta in base al tipo di grafico."""

    try:

        if tipo == TIPO_AUTOMATICO:
            return crea_grafico(df, colonna_x, gruppo)

        return crea_grafico_due_variabili(
            df,
            tipo,
            colonna_x,
            colonna_y,
            gruppo,
            aggregazione
        )

    except Exception as errore:

        return figura_vuota(
            f"Impossibile creare il grafico: {errore}"
        )


# ============================================================
# COLORI DEI GRAFICI
# ============================================================
# Ogni elemento colorabile corrisponde a una "traccia" del grafico
# (una per categoria del gruppo, oppure una sola traccia "Colore").

def colore_predefinito(traccia, posizione):
    """Colore con cui Plotly ha disegnato la traccia (formato #rrggbb)."""

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

        if (
            isinstance(candidato, str)
            and candidato.startswith("#")
            and len(candidato) == 7
        ):
            return candidato

    return PALETTE_COLORI[posizione % len(PALETTE_COLORI)]


def applica_colori(fig, colori):
    """Applica ai grafici i colori scelti ({etichetta: "#rrggbb"})."""

    for traccia in fig.data:

        etichetta = traccia.name or "Colore"
        colore = colori.get(etichetta)

        if colore is None:
            continue

        if traccia.type == "histogram2d":

            fig.update_layout(
                coloraxis_colorscale=[[0, "#ffffff"], [1, colore]]
            )

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

    # Sceglie una variabile iniziale diversa per ogni scheda.
    indice = (numero - 1) % len(colonne)

    colonna_iniziale = colonne[indice]

    # Colori scelti dall'utente per questa scheda: {etichetta: colore}.
    colori_scelti = {}

    # Evita di ridisegnare piu volte mentre si aggiornano i selettori.
    blocco = {"attivo": False}

    selettore_tipo = pn.widgets.Select(
        name="Tipo di grafico",
        options=list(TIPI_GRAFICO.keys()),
        value=TIPO_AUTOMATICO,
        sizing_mode="stretch_width"
    )

    selettore_variabile = pn.widgets.Select(
        name="Variabile",
        options=colonne,
        value=colonna_iniziale,
        sizing_mode="stretch_width"
    )

    opzioni_y = [
        colonna
        for colonna in colonne
        if colonna != colonna_iniziale
    ]

    colonna_y_iniziale = colonne[(indice + 1) % len(colonne)]

    if colonna_y_iniziale not in opzioni_y:
        colonna_y_iniziale = opzioni_y[0] if opzioni_y else None

    selettore_y = pn.widgets.Select(
        name="Variabile Y",
        options=opzioni_y,
        value=colonna_y_iniziale,
        visible=False,
        sizing_mode="stretch_width"
    )

    opzioni_gruppo = [
        "Nessuno"
    ] + [
        colonna
        for colonna in colonne
        if colonna != colonna_iniziale
    ]

    selettore_gruppo = pn.widgets.Select(
        name="Confronta per",
        options=opzioni_gruppo,
        value="Nessuno",
        sizing_mode="stretch_width"
    )

    selettore_aggregazione = pn.widgets.Select(
        name="Aggregazione",
        options=list(FUNZIONI_AGGREGAZIONE.keys()),
        value="Media",
        visible=False,
        sizing_mode="stretch_width"
    )

    grafico = pn.pane.Plotly(
        figura_vuota(""),
        height=320,
        sizing_mode="stretch_width",
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True
        }
    )

    pulsante_ripristina_colori = pn.widgets.Button(
        name="Ripristina colori predefiniti",
        button_type="light"
    )

    scheda_colori = pn.Card(
        title="Colori",
        collapsed=True,
        sizing_mode="stretch_width"
    )

    def cambia_colore(event):

        colori_scelti[event.obj.name] = event.new

        applica_colori(grafico.object, colori_scelti)

        # La figura è stata modificata "sul posto": forza il ridisegno.
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

        colonna_y = None

        if impostazioni["y"]:
            colonna_y = selettore_y.value

        gruppo = None

        if impostazioni["gruppo"] and selettore_gruppo.value != "Nessuno":
            gruppo = selettore_gruppo.value

        fig = crea_grafico_scelto(
            df,
            tipo,
            colonna,
            colonna_y,
            gruppo,
            selettore_aggregazione.value
        )

        

        # Colori predefiniti, letti prima di applicare quelli scelti.
        predefiniti = {}

        for posizione, traccia in enumerate(fig.data):

            etichetta = traccia.name or "Colore"

            if etichetta not in predefiniti:
                predefiniti[etichetta] = colore_predefinito(
                    traccia,
                    posizione
                )

        applica_colori(fig, colori_scelti)

        grafico.object = fig

        # Un selettore di colore per ogni elemento del grafico.
        selettori_colore = []

        for etichetta, predefinito in predefiniti.items():

            selettore = pn.widgets.ColorPicker(
                name=etichetta,
                value=colori_scelti.get(etichetta, predefinito),
                width=130
            )

            selettore.param.watch(
                cambia_colore,
                "value"
            )

            selettori_colore.append(selettore)

        if len(selettori_colore) > 0:

            scheda_colori.objects = [
                pn.GridBox(
                    *selettori_colore,
                    ncols=2
                ),
                pulsante_ripristina_colori
            ]

        else:

            scheda_colori.objects = [
                pn.pane.Markdown("_Nessun elemento da colorare._")
            ]

    def cambia_variabili(event=None):

        if blocco["attivo"]:
            return

        blocco["attivo"] = True

        try:

            tipo = selettore_tipo.value
            impostazioni = TIPI_GRAFICO[tipo]

            usa_y = impostazioni["y"]

            selettore_variabile.name = (
                "Variabile X" if usa_y else "Variabile"
            )

            selettore_y.visible = usa_y
            selettore_gruppo.visible = impostazioni["gruppo"]
            selettore_aggregazione.visible = impostazioni["aggregazione"]

            colonna = selettore_variabile.value

            nuove_opzioni_y = [
                nome
                for nome in colonne
                if nome != colonna
            ]

            selettore_y.options = nuove_opzioni_y

            if selettore_y.value not in nuove_opzioni_y:

                if len(nuove_opzioni_y) > 0:
                    selettore_y.value = nuove_opzioni_y[0]
                else:
                    selettore_y.value = None

            escluse = [colonna]

            if usa_y:
                escluse.append(selettore_y.value)

            nuove_opzioni_gruppo = [
                "Nessuno"
            ] + [
                nome
                for nome in colonne
                if nome not in escluse
            ]

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

    # Primo disegno.
    ridisegna()

    return pn.Column(
        pn.pane.Markdown(
            f"### Grafico {numero}"
        ),
        selettore_tipo,
        pn.Row(
            selettore_variabile,
            selettore_y,
            sizing_mode="stretch_width"
        ),
        pn.Row(
            selettore_gruppo,
            selettore_aggregazione,
            sizing_mode="stretch_width"
        ),
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

        return pn.pane.Markdown(
            "Il dataset non contiene colonne da visualizzare."
        )

    schede = []

    for numero in range(
        1,
        NUMERO_GRAFICI + 1
    ):

        scheda = crea_scheda_grafico(
            df,
            colonne,
            numero
        )

        schede.append(scheda)

    return pn.GridBox(
        *schede,
        ncols=3,
        sizing_mode="stretch_width"
    )


# ============================================================
# COMPONENTI PRINCIPALI DELLA DASHBOARD
# ============================================================

file_input = pn.widgets.FileInput(
    name="Carica CSV",
    accept=".csv",
    sizing_mode="stretch_width"
)

messaggio = pn.pane.Markdown(
    "## Carica un dataset CSV\n\n"
    "Seleziona un file dalla barra laterale per iniziare."
)

nome_dataset = pn.pane.Markdown("")

# Contenuto della prima scheda (Esplorazione).
esplorazione = pn.Column(
    sizing_mode="stretch_width"
)

main = pn.Column(
    messaggio,
    sizing_mode="stretch_width"
)


# ============================================================
# SCHEDA "PULIZIA DATI": COMPONENTI
# ============================================================

# ---------- Barra di stato (sempre visibile) ----------

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


# ---------- Duplicati ----------

dup_colonne = pn.widgets.MultiSelect(
    name="Colonne da confrontare",
    options=[],
    size=8,
    sizing_mode="stretch_width"
)

dup_mantieni = pn.widgets.Select(
    name="Da ogni gruppo di duplicati mantieni",
    options=["Prima occorrenza", "Ultima occorrenza"],
    value="Prima occorrenza",
    sizing_mode="stretch_width"
)

dup_cerca = pn.widgets.Button(
    name="Cerca duplicati",
    button_type="primary"
)

dup_rimuovi = pn.widgets.Button(
    name="Rimuovi duplicati…",
    button_type="danger",
    visible=False
)

dup_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

dup_risultato = pn.Column(sizing_mode="stretch_width")

dup_conferma, dup_chiedi, dup_nascondi = crea_area_conferma()


# ---------- Valori mancanti ----------

mis_statistiche = pn.Column(sizing_mode="stretch_width")

mis_colonne = pn.widgets.MultiSelect(
    name="Colonne da considerare",
    options=[],
    size=8,
    sizing_mode="stretch_width"
)

mis_mostra = pn.widgets.Button(
    name="Mostra righe interessate",
    button_type="primary"
)

mis_rimuovi = pn.widgets.Button(
    name="Rimuovi righe…",
    button_type="danger"
)

mis_metodo = pn.widgets.Select(
    name="Imputa i valori mancanti con",
    options=["Media", "Mediana", "Moda", "Valore personalizzato"],
    value="Media",
    sizing_mode="stretch_width"
)

mis_valore = pn.widgets.TextInput(
    name="Valore personalizzato",
    placeholder="Es. 0, N/D, Sconosciuto",
    visible=False,
    sizing_mode="stretch_width"
)

mis_imputa = pn.widgets.Button(
    name="Imputa valori",
    button_type="success"
)

mis_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

mis_risultato = pn.Column(sizing_mode="stretch_width")

mis_conferma, mis_chiedi, mis_nascondi = crea_area_conferma()


# ---------- Outlier ----------

out_colonna = pn.widgets.Select(
    name="Variabile numerica",
    options=[],
    sizing_mode="stretch_width"
)

out_metodo = pn.widgets.Select(
    name="Metodo",
    options=["IQR (1,5 × IQR)", "Percentili 5° e 95°"],
    value="IQR (1,5 × IQR)",
    sizing_mode="stretch_width"
)

out_cerca = pn.widgets.Button(
    name="Individua outlier",
    button_type="primary"
)

out_rimuovi = pn.widgets.Button(
    name="Rimuovi outlier…",
    button_type="danger",
    visible=False
)

out_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

out_risultato = pn.Column(sizing_mode="stretch_width")

out_conferma, out_chiedi, out_nascondi = crea_area_conferma()


# ---------- Colonne ----------

col_informazioni = pn.Column(sizing_mode="stretch_width")

col_selezione = pn.widgets.MultiSelect(
    name="Colonne da rimuovere",
    options=[],
    size=10,
    sizing_mode="stretch_width"
)

col_rimuovi = pn.widgets.Button(
    name="Rimuovi colonne selezionate…",
    button_type="danger"
)

col_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

col_conferma, col_chiedi, col_nascondi = crea_area_conferma()


# ---------- Dataset pulito (riepilogo, ripristino, download) ----------

riepilogo_dataset = pn.pane.Markdown("")

registro_operazioni = pn.pane.Markdown("")

ripristina = pn.widgets.Button(
    name="Ripristina dataset originale…",
    button_type="warning"
)

ripristino_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

ripristino_conferma, ripristino_chiedi, ripristino_nascondi = (
    crea_area_conferma()
)

selettore_separatore = pn.widgets.Select(
    name="Separatore del CSV scaricato",
    options={
        "Virgola ( , )": ",",
        "Punto e virgola ( ; )": ";"
    },
    value=",",
    width=250
)


def genera_csv():
    """Crea il CSV del dataset pulito. Non tocca il file originale."""

    df = stato["pulito"]

    if df is None:
        return None

    testo = df.to_csv(
        index=False,
        sep=selettore_separatore.value
    )

    # utf-8-sig: Excel riconosce correttamente le lettere accentate.
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
    """
    Riallinea la scheda di pulizia al dataset di lavoro corrente:
    opzioni dei selettori, statistiche e riepiloghi.
    Cancella anche i risultati precedenti, che non sarebbero piu validi.
    """

    df = stato["pulito"]
    originale = stato["originale"]

    if df is None or originale is None:
        return

    colonne = list(df.columns)
    maschera = maschera_mancanti(df)

    colonne_con_mancanti = [
        colonna
        for colonna in colonne
        if maschera[colonna].any()
    ]

    colonne_numeriche = list(
        df.select_dtypes(include="number").columns
    )

    # ----- Barra di stato -----

    righe_rimosse = len(originale) - len(df)

    testo_stato_pulizia.object = (
        f"**Dataset originale:** {originale.shape[0]} righe × "
        f"{originale.shape[1]} colonne &nbsp;|&nbsp; "
        f"**Dataset pulito (copia di lavoro):** {df.shape[0]} righe × "
        f"{df.shape[1]} colonne &nbsp;|&nbsp; "
        f"**Righe rimosse:** {righe_rimosse}"
    )

    # ----- Duplicati -----

    dup_colonne.value = []
    dup_colonne.options = colonne
    dup_colonne.value = colonne

    dup_nascondi()
    dup_esito.visible = False
    dup_rimuovi.visible = False
    dup_risultato.objects = []

    # ----- Valori mancanti -----

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
        "Mancanti": [int(conteggi[colonna]) for colonna in colonne]
    })

    if righe > 0:
        tabella_mancanti["Mancanti %"] = (
            tabella_mancanti["Mancanti"] / righe * 100
        ).round(2)
    else:
        tabella_mancanti["Mancanti %"] = 0.0

    tabella_mancanti = tabella_mancanti.sort_values(
        "Mancanti",
        ascending=False
    )

    totale_mancanti = int(maschera.sum().sum())
    righe_incomplete = int(maschera.any(axis=1).sum())

    mis_statistiche.objects = [
        pn.pane.Markdown(
            f"**{totale_mancanti}** valori mancanti in "
            f"**{righe_incomplete}** righe incomplete "
            f"(su {righe} righe totali)."
        ),
        crea_tabella(
            tabella_mancanti,
            con_indice=False
        )
    ]

    # ----- Outlier -----

    out_colonna.options = colonne_numeriche

    if len(colonne_numeriche) > 0:
        out_colonna.value = colonne_numeriche[0]
    else:
        out_colonna.value = None

    out_cerca.disabled = len(colonne_numeriche) == 0

    out_nascondi()
    out_esito.visible = False
    out_rimuovi.visible = False
    out_risultato.objects = []

    if len(colonne_numeriche) == 0:
        mostra_esito(
            out_esito,
            "Il dataset non contiene colonne numeriche.",
            "warning"
        )

    # ----- Colonne -----

    tabella_colonne = pd.DataFrame({
        "Colonna": colonne,
        "Tipo": [str(tipo) for tipo in df.dtypes],
        "Valori distinti": [
            int(df[colonna].nunique())
            for colonna in colonne
        ],
        "Mancanti": [int(conteggi[colonna]) for colonna in colonne],
        "Mancanti %": list(tabella_mancanti.set_index("Colonna")
                           .loc[colonne, "Mancanti %"])
    })

    col_informazioni.objects = [
        crea_tabella(
            tabella_colonne,
            con_indice=False
        )
    ]

    col_selezione.value = []
    col_selezione.options = colonne

    col_nascondi()
    col_esito.visible = False

    # ----- Dataset pulito -----

    riepilogo_dataset.object = (
        "| | Originale | Dataset pulito |\n"
        "|---|---:|---:|\n"
        f"| Righe | {originale.shape[0]} | {df.shape[0]} |\n"
        f"| Colonne | {originale.shape[1]} | {df.shape[1]} |\n"
        f"| Valori mancanti | "
        f"{int(maschera_mancanti(originale).sum().sum())} | "
        f"{totale_mancanti} |\n"
        f"| Righe duplicate (tutte le colonne) | "
        f"{int(originale.duplicated().sum())} | "
        f"{int(df.duplicated().sum())} |\n"
    )

    if len(stato["registro"]) == 0:
        registro_operazioni.object = (
            "_Nessuna operazione eseguita: il dataset pulito "
            "coincide con l'originale._"
        )
    else:
        elenco = [
            f"{numero}. {voce}"
            for numero, voce in enumerate(stato["registro"], start=1)
        ]

        registro_operazioni.object = "\n".join(elenco)

    ripristino_nascondi()
    ripristino_esito.visible = False


def aggiorna_esplorazione():
    """Ricostruisce la scheda Esplorazione usando il dataset di lavoro."""

    df = stato["pulito"]
    originale = stato["originale"]

    if df is None or originale is None:
        return

    nuove_statistiche = crea_statistiche(df)
    nuova_tabella = crea_tabella_statistiche(df)
    nuovi_grafici = crea_istogrammi(df)

    nome_file = stato["nome_file"]

    nome_dataset.object = (
        f"### Dataset caricato: `{nome_file}`\n\n"
        f"Dimensioni originali: **{originale.shape[0]} righe** × "
        f"**{originale.shape[1]} colonne** &nbsp;|&nbsp; "
        f"Dataset di lavoro mostrato: **{df.shape[0]} righe** × "
        f"**{df.shape[1]} colonne**"
    )

    esplorazione.objects = [

        pn.pane.Markdown(
            "## Panoramica del dataset"
        ),

        nuove_statistiche,

        pn.layout.Divider(),

        pn.pane.Markdown(
            "## Statistiche descrittive"
        ),

        nuova_tabella,

        pn.layout.Divider(),

        pn.pane.Markdown(
            "## Esplorazione delle variabili"
        ),

        nuovi_grafici

    ]


def aggiorna_tutto():
    aggiorna_esplorazione()
    aggiorna_pulizia()
    aggiorna_modelli()


def applica_modifica(nuovo_df, descrizione):
    """
    Sostituisce la copia di lavoro con il nuovo DataFrame,
    registra l'operazione e aggiorna l'interfaccia.
    Il dataset originale non viene toccato.
    """

    stato["pulito"] = nuovo_df
    stato["registro"].append(descrizione)

    aggiorna_tutto()


# ============================================================
# PULIZIA: DUPLICATI
# ============================================================

def valore_keep():
    if dup_mantieni.value == "Prima occorrenza":
        return "first"

    return "last"


def cerca_duplicati(event=None):

    dup_nascondi()
    dup_rimuovi.visible = False
    dup_risultato.objects = []

    df = stato["pulito"]

    if df is None:
        return

    colonne = list(dup_colonne.value)

    if len(colonne) == 0:
        mostra_esito(
            dup_esito,
            "Seleziona almeno una colonna da confrontare.",
            "warning"
        )
        return

    keep = valore_keep()

    # Tutte le righe che fanno parte di un gruppo di duplicati.
    in_gruppi = df.duplicated(subset=colonne, keep=False)

    # Solo le righe che verrebbero eliminate.
    da_rimuovere = df.duplicated(subset=colonne, keep=keep)

    numero_in_gruppi = int(in_gruppi.sum())
    numero_da_rimuovere = int(da_rimuovere.sum())

    if numero_in_gruppi == 0:
        mostra_esito(
            dup_esito,
            "Nessun duplicato trovato con le colonne selezionate.",
            "success"
        )
        return

    visibili = df[in_gruppi].copy()

    visibili.insert(
        0,
        "Azione",
        np.where(
            da_rimuovere[in_gruppi],
            "Da rimuovere",
            "Da mantenere"
        )
    )

    # Ordina in modo che le righe uguali risultino vicine.
    try:
        visibili = visibili.sort_values(
            by=colonne,
            kind="stable"
        )
    except Exception:
        pass

    dup_risultato.objects = [
        crea_tabella(visibili)
    ]

    dup_rimuovi.visible = True

    mostra_esito(
        dup_esito,
        f"Trovate **{numero_in_gruppi}** righe coinvolte in gruppi di "
        f"duplicati. Ne verrebbero rimosse **{numero_da_rimuovere}**.",
        "info"
    )


def chiedi_rimozione_duplicati(event=None):

    df = stato["pulito"]

    if df is None:
        return

    # Colonne e criterio vengono "fissati" nel momento della richiesta.
    colonne = list(dup_colonne.value)
    keep = valore_keep()

    if len(colonne) == 0:
        return

    numero = int(df.duplicated(subset=colonne, keep=keep).sum())

    if numero == 0:
        return

    def esegui():

        nuovo_df = stato["pulito"].drop_duplicates(
            subset=colonne,
            keep=keep
        ).copy()

        rimosse = len(stato["pulito"]) - len(nuovo_df)

        applica_modifica(
            nuovo_df,
            f"Duplicati: rimosse {rimosse} righe "
            f"(colonne confrontate: {', '.join(colonne)}; "
            f"mantenuta la {'prima' if keep == 'first' else 'ultima'} "
            f"occorrenza)."
        )

        mostra_esito(
            dup_esito,
            f"Rimosse **{rimosse}** righe duplicate.",
            "success"
        )

    dup_chiedi(
        f"Stai per rimuovere **{numero}** righe duplicate dal dataset "
        f"pulito. Il dataset originale non viene modificato e potrai "
        f"ripristinarlo. Confermi?",
        esegui
    )


dup_cerca.on_click(cerca_duplicati)
dup_rimuovi.on_click(chiedi_rimozione_duplicati)


# ============================================================
# PULIZIA: VALORI MANCANTI
# ============================================================

def mostra_righe_mancanti(event=None):

    mis_nascondi()
    mis_risultato.objects = []

    df = stato["pulito"]

    if df is None:
        return

    colonne = list(mis_colonne.value)

    if len(colonne) == 0:
        mostra_esito(
            mis_esito,
            "Seleziona almeno una colonna.",
            "warning"
        )
        return

    maschera = maschera_mancanti(df)[colonne]
    righe_interessate = maschera.any(axis=1)

    totale = int(righe_interessate.sum())

    if totale == 0:
        mostra_esito(
            mis_esito,
            "Nessun valore mancante nelle colonne selezionate.",
            "success"
        )
        return

    maschera_visibile = maschera[righe_interessate].head(MAX_RIGHE_TABELLA)

    visibili = df.loc[maschera_visibile.index].copy()

    visibili.insert(
        0,
        "Colonne mancanti",
        maschera_visibile.apply(
            lambda riga: ", ".join(riga.index[riga.values]),
            axis=1
        )
    )

    mis_risultato.objects = [
        crea_tabella(
            visibili,
            totale=totale
        )
    ]

    mostra_esito(
        mis_esito,
        f"**{totale}** righe hanno almeno un valore mancante "
        f"nelle colonne selezionate.",
        "info"
    )


def chiedi_rimozione_mancanti(event=None):

    df = stato["pulito"]

    if df is None:
        return

    colonne = list(mis_colonne.value)

    if len(colonne) == 0:
        mostra_esito(
            mis_esito,
            "Seleziona almeno una colonna.",
            "warning"
        )
        return

    righe_interessate = maschera_mancanti(df)[colonne].any(axis=1)
    numero = int(righe_interessate.sum())

    if numero == 0:
        mostra_esito(
            mis_esito,
            "Nessuna riga da rimuovere nelle colonne selezionate.",
            "success"
        )
        return

    def esegui():

        corrente = stato["pulito"]

        da_tenere = ~maschera_mancanti(corrente)[colonne].any(axis=1)
        nuovo_df = corrente.loc[da_tenere].copy()

        rimosse = len(corrente) - len(nuovo_df)

        applica_modifica(
            nuovo_df,
            f"Valori mancanti: rimosse {rimosse} righe "
            f"(colonne considerate: {', '.join(colonne)})."
        )

        mostra_esito(
            mis_esito,
            f"Rimosse **{rimosse}** righe con valori mancanti.",
            "success"
        )

    mis_chiedi(
        f"Stai per rimuovere **{numero}** righe con valori mancanti "
        f"dal dataset pulito. Confermi?",
        esegui
    )


def imputa_valori_mancanti(event=None):

    mis_nascondi()

    df_corrente = stato["pulito"]

    if df_corrente is None:
        return

    colonne = list(mis_colonne.value)
    metodo = mis_metodo.value

    if len(colonne) == 0:
        mostra_esito(
            mis_esito,
            "Seleziona almeno una colonna.",
            "warning"
        )
        return

    testo_personalizzato = mis_valore.value.strip()

    if metodo == "Valore personalizzato" and testo_personalizzato == "":
        mostra_esito(
            mis_esito,
            "Scrivi il valore personalizzato da usare.",
            "warning"
        )
        return

    # Lavora su una copia: il dataset corrente cambia solo a fine operazione.
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
                    valore = float(
                        testo_personalizzato.replace(",", ".")
                    )
                except ValueError:
                    colonne_saltate.append(
                        f"{colonna} (il valore non è un numero)"
                    )
                    continue

            else:
                valore = testo_personalizzato

        if pd.isna(valore):
            colonne_saltate.append(f"{colonna} (nessun valore valido)")
            continue

        try:
            df.loc[mancanti, colonna] = valore
            valori_sostituiti += numero

        except Exception:
            colonne_saltate.append(f"{colonna} (tipo non compatibile)")

    testo_saltate = ""

    if len(colonne_saltate) > 0:
        testo_saltate = (
            "\n\nColonne saltate: " + "; ".join(colonne_saltate) + "."
        )

    if valori_sostituiti == 0:
        mostra_esito(
            mis_esito,
            "Nessun valore è stato imputato." + testo_saltate,
            "warning"
        )
        return

    applica_modifica(
        df,
        f"Valori mancanti: imputati {valori_sostituiti} valori "
        f"con il metodo «{metodo}» "
        f"(colonne: {', '.join(colonne)})."
    )

    mostra_esito(
        mis_esito,
        f"Imputati **{valori_sostituiti}** valori con il metodo "
        f"«{metodo}»." + testo_saltate,
        "success"
    )


def aggiorna_visibilita_valore(event=None):
    mis_valore.visible = (
        mis_metodo.value == "Valore personalizzato"
    )


mis_mostra.on_click(mostra_righe_mancanti)
mis_rimuovi.on_click(chiedi_rimozione_mancanti)
mis_imputa.on_click(imputa_valori_mancanti)

mis_metodo.param.watch(
    aggiorna_visibilita_valore,
    "value"
)


# ============================================================
# PULIZIA: OUTLIER
# ============================================================

def calcola_outlier(serie, metodo):
    """
    Restituisce (maschera, limite_inferiore, limite_superiore).
    I valori mancanti non sono mai considerati outlier.
    """

    valide = serie.dropna()

    if metodo.startswith("IQR"):

        q1 = valide.quantile(0.25)
        q3 = valide.quantile(0.75)
        iqr = q3 - q1

        limite_inferiore = q1 - 1.5 * iqr
        limite_superiore = q3 + 1.5 * iqr

    else:

        limite_inferiore = valide.quantile(0.05)
        limite_superiore = valide.quantile(0.95)

    maschera = (
        (serie < limite_inferiore)
        | (serie > limite_superiore)
    )

    return maschera, limite_inferiore, limite_superiore


def crea_grafico_outlier(
    df,
    colonna,
    maschera,
    limite_inferiore,
    limite_superiore,
    metodo
):

    dati = pd.DataFrame({
        "Indice riga": df.index,
        colonna: df[colonna].values,
        "Tipo": np.where(
            maschera.values,
            "Outlier",
            "Valore normale"
        )
    })

    dati = dati.dropna(subset=[colonna])

    normali = dati[dati["Tipo"] == "Valore normale"]
    outlier = dati[dati["Tipo"] == "Outlier"]

    nota = ""

    if len(normali) > MAX_PUNTI_OUTLIER:
        normali = normali.sample(
            n=MAX_PUNTI_OUTLIER,
            random_state=42
        )

        nota = f" (valori normali: campione di {MAX_PUNTI_OUTLIER})"

    dati = pd.concat([normali, outlier])

    fig = px.scatter(
        dati,
        x="Indice riga",
        y=colonna,
        color="Tipo",
        color_discrete_map={
            "Valore normale": "#2563eb",
            "Outlier": "#dc2626"
        },
        category_orders={
            "Tipo": ["Valore normale", "Outlier"]
        },
        title=f"Outlier di {colonna} – {metodo}{nota}"
    )

    fig.add_hline(
        y=limite_inferiore,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Limite inferiore"
    )

    fig.add_hline(
        y=limite_superiore,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text="Limite superiore"
    )

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        margin=dict(l=40, r=20, t=60, b=60),
        font=dict(size=11),
        legend_title_text=""
    )

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
        mostra_esito(
            out_esito,
            f"La colonna «{colonna}» non contiene valori validi.",
            "warning"
        )
        return

    metodo = out_metodo.value

    maschera, limite_inferiore, limite_superiore = calcola_outlier(
        serie,
        metodo
    )

    numero = int(maschera.sum())

    grafico = pn.pane.Plotly(
        crea_grafico_outlier(
            df,
            colonna,
            maschera,
            limite_inferiore,
            limite_superiore,
            metodo
        ),
        height=380,
        sizing_mode="stretch_width",
        config={
            "responsive": True,
            "displayModeBar": True,
            "scrollZoom": True
        }
    )

    oggetti = [grafico]

    if numero > 0:
        oggetti.append(
            crea_tabella(df[maschera])
        )

        out_rimuovi.visible = True

    out_risultato.objects = oggetti

    mostra_esito(
        out_esito,
        f"Limiti: da **{limite_inferiore:.4g}** a "
        f"**{limite_superiore:.4g}**. "
        f"Outlier trovati: **{numero}** su {len(serie.dropna())} "
        f"valori validi.",
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

        maschera_attuale, _, _ = calcola_outlier(
            corrente[colonna],
            metodo
        )

        nuovo_df = corrente.loc[~maschera_attuale].copy()

        rimosse = len(corrente) - len(nuovo_df)

        applica_modifica(
            nuovo_df,
            f"Outlier: rimosse {rimosse} righe "
            f"(colonna: {colonna}; metodo: {metodo})."
        )

        mostra_esito(
            out_esito,
            f"Rimosse **{rimosse}** righe con outlier in «{colonna}».",
            "success"
        )

    out_chiedi(
        f"Stai per rimuovere **{numero}** righe con outlier nella "
        f"colonna «{colonna}» ({metodo}). Confermi?",
        esegui
    )


out_cerca.on_click(individua_outlier)
out_rimuovi.on_click(chiedi_rimozione_outlier)


# ============================================================
# PULIZIA: RIMOZIONE DI COLONNE
# ============================================================

def chiedi_rimozione_colonne(event=None):

    df = stato["pulito"]

    if df is None:
        return

    colonne = list(col_selezione.value)

    if len(colonne) == 0:
        mostra_esito(
            col_esito,
            "Seleziona almeno una colonna da rimuovere.",
            "warning"
        )
        return

    if len(colonne) >= len(df.columns):
        mostra_esito(
            col_esito,
            "Non puoi rimuovere tutte le colonne: ne deve restare "
            "almeno una.",
            "warning"
        )
        return

    def esegui():

        nuovo_df = stato["pulito"].drop(columns=colonne).copy()

        applica_modifica(
            nuovo_df,
            f"Colonne: rimosse {len(colonne)} colonne "
            f"({', '.join(colonne)})."
        )

        mostra_esito(
            col_esito,
            f"Rimosse **{len(colonne)}** colonne: "
            f"{', '.join(colonne)}.",
            "success"
        )

    col_chiedi(
        f"Stai per rimuovere **{len(colonne)}** colonne dal dataset "
        f"pulito: {', '.join(colonne)}. L'originale non viene "
        f"modificato e potrai ripristinarlo. Confermi?",
        esegui
    )


col_rimuovi.on_click(chiedi_rimozione_colonne)


# ============================================================
# PULIZIA: RIPRISTINO DEL DATASET ORIGINALE
# ============================================================

def chiedi_ripristino(event=None):

    if stato["originale"] is None:
        return

    if len(stato["registro"]) == 0:
        mostra_esito(
            ripristino_esito,
            "Non ci sono modifiche da annullare.",
            "info"
        )
        return

    def esegui():

        stato["pulito"] = stato["originale"].copy()
        stato["registro"] = []

        aggiorna_tutto()

        mostra_esito(
            ripristino_esito,
            "Dataset originale ripristinato.",
            "success"
        )

    ripristino_chiedi(
        "Il dataset pulito tornerà identico all'originale e tutte le "
        "modifiche eseguite andranno perse. Confermi?",
        esegui
    )


ripristina.on_click(chiedi_ripristino)


# ============================================================
# SCHEDA "MODELLI": FUNZIONI
# ============================================================

def crea_preprocessore(colonne_numeriche, colonne_categoriche, scala):
    """
    Preparazione dei dati, appresa solo sul train set (niente data leakage):
    - numeriche: mediana per i mancanti (+ standardizzazione se serve)
    - categoriche: valore piu frequente per i mancanti + one-hot encoding
    """

    passi_numerici = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if scala:
        passi_numerici.append(("scaler", StandardScaler()))

    trasformatori = []

    if len(colonne_numeriche) > 0:
        trasformatori.append(
            ("num", Pipeline(passi_numerici), colonne_numeriche)
        )

    if len(colonne_categoriche) > 0:
        trasformatori.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
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
    """

    avvisi = []

    feature = [colonna for colonna in feature if colonna != target]

    if len(feature) == 0:
        raise ValueError("Seleziona almeno una variabile esplicativa.")

    # Righe con target mancante: non utilizzabili.
    target_mancante = maschera_mancanti(df[[target]])[target]

    if target_mancante.any():
        avvisi.append(
            f"Escluse {int(target_mancante.sum())} righe con target mancante."
        )

    dati = df.loc[~target_mancante]

    if dati.empty:
        raise ValueError("Nessuna riga con target valido.")

    codificatore = LabelEncoder()

    y = codificatore.fit_transform(dati[target].astype(str))

    classi = [str(classe) for classe in codificatore.classes_]

    if len(classi) < 2:
        raise ValueError("Il target deve avere almeno 2 classi.")

    if len(classi) > MAX_CLASSI:
        raise ValueError(
            f"Il target ha {len(classi)} classi (massimo {MAX_CLASSI})."
        )

    X = dati[feature].copy()
    mancanti = maschera_mancanti(X)

    colonne_numeriche = []
    colonne_categoriche = []
    escluse = []

    for colonna in feature:

        if pd.api.types.is_numeric_dtype(X[colonna]):

            X[colonna] = X[colonna].astype(float)

            if X[colonna].isna().all():
                escluse.append(f"{colonna} (tutti i valori mancanti)")
                continue

            colonne_numeriche.append(colonna)

        else:

            X[colonna] = (
                X[colonna]
                .astype(object)
                .where(~mancanti[colonna], np.nan)
            )

            X[colonna] = X[colonna].map(
                lambda valore: valore if pd.isna(valore) else str(valore)
            )

            if X[colonna].isna().all():
                escluse.append(f"{colonna} (tutti i valori mancanti)")
                continue

            if X[colonna].nunique() > MAX_CATEGORIE_FEATURE:
                escluse.append(
                    f"{colonna} (oltre {MAX_CATEGORIE_FEATURE} categorie)"
                )
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
    """Restituisce (modello, serve_standardizzazione)."""

    if nome == "Regressione logistica":

        return (
            LogisticRegression(
                C=parametri["lr_c"],
                max_iter=1000,
                random_state=seed
            ),
            True
        )

    if nome == "Albero decisionale":

        return (
            DecisionTreeClassifier(
                max_depth=parametri["dt_profondita"],
                random_state=seed
            ),
            False
        )

    if nome == "Random forest":

        return (
            RandomForestClassifier(
                n_estimators=parametri["rf_alberi"],
                max_depth=parametri["rf_profondita"],
                random_state=seed,
                n_jobs=-1
            ),
            False
        )

    if nome == "CatBoost":

        if not CATBOOST_DISPONIBILE:
            raise RuntimeError(
                "CatBoost non è installato (pip install catboost)."
            )

        return (
            CatBoostClassifier(
                iterations=parametri["cb_iterazioni"],
                depth=parametri["cb_profondita"],
                learning_rate=parametri["cb_learning_rate"],
                random_seed=seed,
                verbose=0,
                allow_writing_files=False
            ),
            False
        )

    if nome == "K-nearest neighbors":

        return (
            KNeighborsClassifier(
                n_neighbors=max(1, min(parametri["knn_k"], numero_train))
            ),
            True
        )

    raise ValueError(f"Modello sconosciuto: {nome}")


def nomi_feature(pipeline):
    """Nomi delle colonne dopo la preparazione (one-hot incluso)."""

    nomi = pipeline.named_steps["prep"].get_feature_names_out()

    return [str(nome).split("__", 1)[1] for nome in nomi]


def calcola_importanze(pipeline, X_test, y_test, seed):
    """Restituisce (Series con le importanze, titolo del grafico)."""

    modello = pipeline.named_steps["modello"]

    if hasattr(modello, "feature_importances_"):

        valori = np.asarray(modello.feature_importances_, dtype=float)

        return (
            pd.Series(valori, index=nomi_feature(pipeline)),
            "Importanza delle variabili"
        )

    if hasattr(modello, "coef_"):

        valori = np.abs(
            np.asarray(modello.coef_, dtype=float)
        ).mean(axis=0)

        return (
            pd.Series(valori, index=nomi_feature(pipeline)),
            "Importanza delle variabili (|coefficiente|, dati standardizzati)"
        )

    # Modelli senza importanze interne (KNN): importanza per permutazione.
    campione = X_test
    y_campione = y_test

    if len(X_test) > 1000:

        posizioni = np.random.RandomState(seed).choice(
            len(X_test),
            1000,
            replace=False
        )

        campione = X_test.iloc[posizioni]
        y_campione = y_test[posizioni]

    risultato = permutation_importance(
        pipeline,
        campione,
        y_campione,
        n_repeats=3,
        random_state=seed,
        scoring="accuracy"
    )

    return (
        pd.Series(
            risultato.importances_mean,
            index=list(X_test.columns)
        ),
        "Importanza per permutazione (calo di accuracy)"
    )


def addestra_modelli(df, config):
    """
    Divide il dataset in train e test, addestra i modelli scelti e
    calcola metriche e grafici. Non usa componenti di Panel, quindi
    puo girare in un thread separato.
    """

    (
        X,
        y,
        classi,
        colonne_numeriche,
        colonne_categoriche,
        escluse,
        avvisi
    ) = prepara_dati_modello(
        df,
        config["target"],
        config["feature"]
    )

    if len(X) < 10:
        raise ValueError("Servono almeno 10 righe con target valido.")

    # --- Split train / test ---
    stratificazione = y if config["stratifica"] else None

    try:

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=config["seed"],
            stratify=stratificazione
        )

    except ValueError:

        if stratificazione is None:
            raise

        avvisi.append(
            "Stratificazione non possibile (alcune classi hanno troppo "
            "pochi esempi): usato uno split casuale semplice."
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config["test_size"],
            random_state=config["seed"]
        )

    labels = list(range(len(classi)))

    righe = []
    dettagli = {}
    errori = {}

    for nome in config["modelli"]:
            try:
                stimatore, scala = crea_modello(
                    nome,
                    config["parametri"],
                    config["seed"],
                    len(X_train)
                )

                pipeline = Pipeline([
                    (
                        "prep",
                        crea_preprocessore(
                            colonne_numeriche,
                            colonne_categoriche,
                            scala
                        )
                    ),
                    ("modello", stimatore)
                ])

                inizio = time.perf_counter()
                pipeline.fit(X_train, y_train)
                durata = time.perf_counter() - inizio

                # --- SOGLIA DI DECISIONE PERSONALIZZATA ---
                if len(classi) == 2 and hasattr(pipeline, "predict_proba"):
                    soglia = config.get("soglia", 0.5)

                    prob_train = pipeline.predict_proba(X_train)[:, 1]
                    y_pred_train = (prob_train >= soglia).astype(int)

                    prob_test = pipeline.predict_proba(X_test)[:, 1]
                    y_pred = (prob_test >= soglia).astype(int)
                else:
                    y_pred_train = np.asarray(pipeline.predict(X_train)).ravel()
                    y_pred = np.asarray(pipeline.predict(X_test)).ravel()

                # --- RICALCOLO METRICHE ---
                acc_train = accuracy_score(y_train, y_pred_train)
                acc_test = accuracy_score(y_test, y_pred)

                precisione, richiamo, f1, _ = precision_recall_fscore_support(
                    y_test,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )

                # --- ROC AUC e curva ROC ---
                auc = np.nan
                curva_roc = None
                try:
                    probabilita = pipeline.predict_proba(X_test)
                    if len(classi) == 2:
                        auc = roc_auc_score(y_test, probabilita[:, 1])
                        fpr, tpr, _ = roc_curve(y_test, probabilita[:, 1])
                        curva_roc = (fpr, tpr, auc)
                    else:
                        auc = roc_auc_score(
                            y_test,
                            probabilita,
                            multi_class="ovr",
                            average="weighted"
                        )
                except Exception:
                    pass

                # --- CROSS-VALIDATION (USA CLONE PER NON CORROMPERE IL MODELLO) ---
                cv_media = np.nan
                cv_dev_std = np.nan
                if config["cv"]:
                    try:
                        punteggi = cross_val_score(
                            clone(pipeline),  # <-- USO OBBLIGATORIO DI CLONE
                            X_train,
                            y_train,
                            cv=5,
                            scoring="accuracy"
                        )
                        cv_media = float(np.mean(punteggi))
                        cv_dev_std = float(np.std(punteggi))
                    except Exception as errore_cv:
                        avvisi.append(
                            f"{nome}: cross-validation non eseguita ({type(errore_cv).__name__})."
                        )

                importanze, titolo_importanze = calcola_importanze(
                    pipeline,
                    X_test,
                    y_test,
                    config["seed"]
                )

                testo_albero = None
                if nome == "Albero decisionale":
                    testo_albero = export_text(
                        pipeline.named_steps["modello"],
                        feature_names=nomi_feature(pipeline),
                        max_depth=4
                    )

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=labels,
                    target_names=classi,
                    output_dict=True,
                    zero_division=0
                )

                report = (
                    pd.DataFrame(report)
                    .transpose()
                    .reset_index()
                    .rename(columns={"index": "Classe"})
                    .round(3)
                )

                riga = {
                    "Modello": nome,
                    "Accuracy train": acc_train,
                    "Accuracy test": acc_test,
                    "Precision": precisione,
                    "Recall": richiamo,
                    "F1": f1,
                    "ROC AUC": auc,
                    "Tempo (s)": durata
                }

                if config["cv"]:
                    riga["CV accuracy (media)"] = cv_media
                    riga["CV (dev. std.)"] = cv_dev_std

                righe.append(riga)

                dettagli[nome] = {
                    "matrice": confusion_matrix(
                        y_test,
                        y_pred,
                        labels=labels
                    ),
                    "importanze": importanze,
                    "titolo_importanze": titolo_importanze,
                    "testo_albero": testo_albero,
                    "report": report,
                    "roc": curva_roc
                }

            except Exception as errore:
                errori[nome] = f"{type(errore).__name__}: {errore}"

    if len(righe) == 0:

        dettaglio_errori = "; ".join(
            f"{nome} → {testo}"
            for nome, testo in errori.items()
        )

        raise ValueError(
            f"Nessun modello addestrato con successo. {dettaglio_errori}"
        )

    tabella = pd.DataFrame(righe).round(4)

    tabella = tabella.sort_values(
        "Accuracy test",
        ascending=False
    ).reset_index(drop=True)

    return {
        "target": config["target"],
        "classi": classi,
        "conteggio_classi": {
            classe: int((y == codice).sum())
            for codice, classe in enumerate(classi)
        },
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
# SCHEDA "MODELLI": COMPONENTI
# ============================================================

risultati_modelli = {"ultimo": None}

mod_target = pn.widgets.Select(
    name="Variabile target (da prevedere)",
    options=[],
    sizing_mode="stretch_width"
)

mod_feature = pn.widgets.MultiSelect(
    name="Variabili esplicative (feature)",
    options=[],
    size=8,
    sizing_mode="stretch_width"
)

mod_test_size = pn.widgets.FloatSlider(
    name="Quota del test set",
    start=0.1,
    end=0.5,
    step=0.05,
    value=0.2,
    sizing_mode="stretch_width"
)

mod_seed = pn.widgets.IntInput(
    name="Random state (riproducibilità)",
    value=42,
    start=0,
    sizing_mode="stretch_width"
)

mod_stratifica = pn.widgets.Checkbox(
    name="Stratifica lo split per classe",
    value=True
)

mod_cv = pn.widgets.Checkbox(
    name="Cross-validation a 5 fold sul train set (più lenta)",
    value=False
)

mod_scelta = pn.widgets.CheckBoxGroup(
    name="Modelli da addestrare",
    options=MODELLI,
    value=list(MODELLI)
)

par_lr_c = pn.widgets.FloatInput(
    name="Regressione logistica: C",
    value=1.0,
    start=0.0001,
    step=0.1
)

par_dt_profondita = pn.widgets.IntInput(
    name="Albero: profondità massima (0 = illimitata)",
    value=5,
    start=0
)

par_rf_alberi = pn.widgets.IntInput(
    name="Random forest: numero di alberi",
    value=200,
    start=10,
    end=2000
)

par_rf_profondita = pn.widgets.IntInput(
    name="Random forest: profondità massima (0 = illimitata)",
    value=0,
    start=0
)

par_cb_iterazioni = pn.widgets.IntInput(
    name="CatBoost: iterazioni",
    value=200,
    start=10,
    end=5000
)

par_cb_profondita = pn.widgets.IntInput(
    name="CatBoost: profondità",
    value=6,
    start=1,
    end=10
)

par_cb_learning_rate = pn.widgets.FloatInput(
    name="CatBoost: learning rate",
    value=0.1,
    start=0.001,
    end=1.0,
    step=0.01
)

par_knn_k = pn.widgets.IntInput(
    name="KNN: numero di vicini",
    value=5,
    start=1
)

mod_soglia = pn.widgets.FloatSlider(
    name="Soglia di decisione (Target Binario)",
    start=0.05,
    end=0.95,
    step=0.05,
    value=0.50,
    sizing_mode="stretch_width"
)

scheda_iperparametri = pn.Card(
    par_lr_c,
    par_dt_profondita,
    par_rf_alberi,
    par_rf_profondita,
    par_cb_iterazioni,
    par_cb_profondita,
    par_cb_learning_rate,
    par_knn_k,
    title="Iperparametri",
    collapsed=True,
    sizing_mode="stretch_width"
)

mod_allena = pn.widgets.Button(
    name="Allena e valuta i modelli",
    button_type="primary",
    sizing_mode="stretch_width"
)

mod_esito = pn.pane.Alert(
    "",
    visible=False,
    sizing_mode="stretch_width"
)

mod_dettaglio = pn.widgets.Select(
    name="Modello da analizzare nel dettaglio",
    options=[],
    sizing_mode="stretch_width"
)

mod_dettaglio_area = pn.Column(sizing_mode="stretch_width")

area_modelli = pn.Column(sizing_mode="stretch_width")


# ============================================================
# SCHEDA "MODELLI": AGGIORNAMENTO E RISULTATI
# ============================================================

def imposta_feature_default(event=None):
    """Seleziona tutte le colonne tranne il target."""

    target = mod_target.value

    mod_feature.value = [
        colonna
        for colonna in mod_feature.options
        if colonna != target
    ]


def aggiorna_modelli():
    """Riallinea la scheda Modelli al dataset di lavoro corrente."""

    risultati_modelli["ultimo"] = None

    mod_esito.visible = False

    df = stato["pulito"]

    if not SKLEARN_DISPONIBILE:

        mod_allena.disabled = True

        area_modelli.objects = [
            pn.pane.Alert(
                "La libreria **scikit-learn** non è installata. "
                "Installala con `pip install scikit-learn catboost` "
                "e riavvia l'app.",
                alert_type="danger"
            )
        ]

        return

    area_modelli.objects = [
        pn.pane.Markdown(
            "_Imposta i parametri a sinistra e premi "
            "«Allena e valuta i modelli»._"
        )
    ]

    mod_dettaglio_area.objects = []

    if df is None:
        return

    colonne = list(df.columns)

    maschera = maschera_mancanti(df)

    # Target possibili: colonne con da 2 a MAX_CLASSI valori distinti.
    candidati = []
    binari = []

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

        mod_target.value = (
            binari[0] if len(binari) > 0 else candidati[0]
        )

    else:

        mod_target.value = None

        mostra_esito(
            mod_esito,
            f"Nessuna colonna adatta come target: servono colonne "
            f"con da 2 a {MAX_CLASSI} valori distinti.",
            "warning"
        )

    imposta_feature_default()

    mod_allena.disabled = len(candidati) == 0


def valore_o(widget, predefinito):
    if widget.value is None:
        return predefinito

    return widget.value


def leggi_configurazione():

    return {
        "target": mod_target.value,
        "feature": list(mod_feature.value),
        "test_size": float(mod_test_size.value),
        "soglia": float(mod_soglia.value),
        "seed": int(valore_o(mod_seed, 42)),
        "stratifica": bool(mod_stratifica.value),
        "cv": bool(mod_cv.value),
        "modelli": list(mod_scelta.value),
        "parametri": {
            "lr_c": float(valore_o(par_lr_c, 1.0)),
            "dt_profondita": int(valore_o(par_dt_profondita, 5)) or None,
            "rf_alberi": int(valore_o(par_rf_alberi, 200)),
            "rf_profondita": int(valore_o(par_rf_profondita, 0)) or None,
            "cb_iterazioni": int(valore_o(par_cb_iterazioni, 200)),
            "cb_profondita": int(valore_o(par_cb_profondita, 6)),
            "cb_learning_rate": float(valore_o(par_cb_learning_rate, 0.1)),
            "knn_k": int(valore_o(par_knn_k, 5)),
        }
    }


async def allena_modelli(event=None):
    """
    Callback asincrona: l'addestramento gira in un thread, cosi
    l'interfaccia resta reattiva e mostra l'indicatore di caricamento.
    """

    df = stato["pulito"]

    if df is None or not SKLEARN_DISPONIBILE:
        return

    config = leggi_configurazione()

    if config["target"] is None:
        mostra_esito(mod_esito, "Seleziona una variabile target.", "warning")
        return

    if len(config["modelli"]) == 0:
        mostra_esito(
            mod_esito,
            "Seleziona almeno un modello.",
            "warning"
        )
        return

    mod_esito.visible = False
    mod_allena.disabled = True
    area_modelli.loading = True

    # Lascia il tempo all'interfaccia di mostrare il caricamento.
    await asyncio.sleep(0.05)

    risultati = None

    try:

        risultati = await asyncio.to_thread(
            addestra_modelli,
            df.copy(),
            config
        )

    except Exception as errore:

        traceback.print_exc()

        mostra_esito(
            mod_esito,
            f"**Errore durante l'addestramento:** {errore}",
            "danger"
        )

    finally:

        area_modelli.loading = False
        mod_allena.disabled = False

    if risultati is not None:
        mostra_risultati_modelli(risultati)


def mostra_risultati_modelli(ris):

    risultati_modelli["ultimo"] = ris

    classi = ris["classi"]
    tabella = ris["tabella"]

    distribuzione = ", ".join(
        f"{classe}: {numero}"
        for classe, numero in ris["conteggio_classi"].items()
    )

    riepilogo = (
        f"**Target:** `{ris['target']}` ({len(classi)} classi — "
        f"{distribuzione})  \n"
        f"**Split:** {ris['numero_train']} righe di train / "
        f"{ris['numero_test']} righe di test  \n"
        f"**Variabili usate ({len(ris['feature_usate'])}):** "
        f"{', '.join(ris['feature_usate'])}"
    )

    if len(ris["escluse"]) > 0:
        riepilogo += (
            f"  \n**Variabili escluse:** {', '.join(ris['escluse'])}"
        )

    oggetti = [pn.pane.Markdown(riepilogo)]

    if len(ris["avvisi"]) > 0:
        oggetti.append(
            pn.pane.Alert(
                "\n".join(f"- {avviso}" for avviso in ris["avvisi"]),
                alert_type="warning",
                sizing_mode="stretch_width"
            )
        )

    if len(ris["errori"]) > 0:
        oggetti.append(
            pn.pane.Alert(
                "Modelli non addestrati:\n" + "\n".join(
                    f"- **{nome}**: {testo}"
                    for nome, testo in ris["errori"].items()
                ),
                alert_type="danger",
                sizing_mode="stretch_width"
            )
        )

    # --- Tabella di confronto ---
    oggetti.append(pn.pane.Markdown("### Confronto tra i modelli"))

    oggetti.append(
        pn.widgets.Tabulator(
            tabella,
            show_index=False,
            disabled=True,
            height=70 + 36 * len(tabella),
            sizing_mode="stretch_width"
        )
    )

    oggetti.append(
        pn.pane.Markdown(
            "_Precision, recall e F1 sono medie pesate per classe. "
            "Una grande differenza tra accuracy su train e su test "
            "indica overfitting._"
        )
    )

    # --- Grafico a barre delle metriche ---
    metriche = tabella.melt(
        id_vars="Modello",
        value_vars=[
            "Accuracy test",
            "Precision",
            "Recall",
            "F1",
            "ROC AUC"
        ],
        var_name="Metrica",
        value_name="Valore"
    ).dropna()

    fig_metriche = px.bar(
        metriche,
        x="Modello",
        y="Valore",
        color="Metrica",
        barmode="group",
        title="Metriche sul test set"
    )

    fig_metriche.update_yaxes(range=[0, 1.05])

    impostazioni_comuni(fig_metriche)

    grafici = [
        pn.pane.Plotly(
            fig_metriche,
            height=380,
            sizing_mode="stretch_width",
            config={"responsive": True}
        )
    ]

    # --- Curve ROC (solo target binari) ---
    if len(classi) == 2:

        fig_roc = go.Figure()

        for nome, dettaglio in ris["modelli"].items():

            if dettaglio["roc"] is None:
                continue

            fpr, tpr, auc = dettaglio["roc"]

            fig_roc.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{nome} (AUC {auc:.3f})"
                )
            )

        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Caso (AUC 0.5)",
                line=dict(dash="dash", color="#999999")
            )
        )

        fig_roc.update_layout(
            title=f"Curve ROC (classe positiva: {classi[1]})",
            xaxis_title="False positive rate",
            yaxis_title="True positive rate"
        )

        impostazioni_comuni(fig_roc)

        grafici.append(
            pn.pane.Plotly(
                fig_roc,
                height=380,
                sizing_mode="stretch_width",
                config={"responsive": True}
            )
        )

    else:

        oggetti.append(
            pn.pane.Markdown(
                "_La curva ROC è mostrata solo per target binari; "
                "la colonna ROC AUC usa il metodo one-vs-rest._"
            )
        )

    oggetti.append(
        pn.Row(*grafici, sizing_mode="stretch_width")
    )

    # --- Dettaglio del singolo modello ---
    oggetti.append(pn.layout.Divider())
    oggetti.append(pn.pane.Markdown("### Dettaglio di un modello"))
    oggetti.append(mod_dettaglio)
    oggetti.append(mod_dettaglio_area)

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

    # --- Matrice di confusione ---
    fig_matrice = px.imshow(
        dettaglio["matrice"],
        x=classi,
        y=classi,
        text_auto=True,
        color_continuous_scale="Blues",
        aspect="auto",
        labels=dict(x="Predetto", y="Reale", color="Conteggio"),
        title=f"Matrice di confusione (test set) – {nome}"
    )

    impostazioni_comuni(fig_matrice)

    # --- Importanza delle variabili ---
    importanze = (
        dettaglio["importanze"]
        .sort_values(ascending=False)
        .head(15)
        .iloc[::-1]
    )

    fig_importanze = px.bar(
        x=importanze.values,
        y=importanze.index,
        orientation="h",
        labels={"x": "Importanza", "y": ""},
        title=f"{dettaglio['titolo_importanze']} – {nome}"
    )

    impostazioni_comuni(fig_importanze)

    oggetti = [
        pn.pane.Markdown(
            f"Accuracy train **{riga['Accuracy train']:.3f}** · "
            f"accuracy test **{riga['Accuracy test']:.3f}** · "
            f"F1 **{riga['F1']:.3f}** · "
            f"tempo di addestramento **{riga['Tempo (s)']:.2f} s**"
        ),
        pn.Row(
            pn.pane.Plotly(
                fig_matrice,
                height=380,
                sizing_mode="stretch_width",
                config={"responsive": True}
            ),
            pn.pane.Plotly(
                fig_importanze,
                height=380,
                sizing_mode="stretch_width",
                config={"responsive": True}
            ),
            sizing_mode="stretch_width"
        ),
        pn.pane.Markdown("#### Report per classe (test set)"),
        pn.widgets.Tabulator(
            dettaglio["report"],
            show_index=False,
            disabled=True,
            height=min(400, 70 + 36 * len(dettaglio["report"])),
            sizing_mode="stretch_width"
        )
    ]

    if dettaglio["testo_albero"] is not None:

        legenda = ", ".join(
            f"{codice} = {classe}"
            for codice, classe in enumerate(classi)
        )

        oggetti.append(
            pn.pane.Markdown(
                f"#### Regole dell'albero (primi 4 livelli)\n"
                f"Codici delle classi: {legenda}\n\n"
                f"```\n{dettaglio['testo_albero']}\n```"
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
            "CatBoost non è installato: il modello CatBoost non sarà "
            "disponibile (`pip install catboost`).",
            alert_type="warning",
            sizing_mode="stretch_width"
        )
    )

scheda_modelli = pn.Column(
    pn.pane.Markdown(
        "### Modelli di classificazione\n"
        "Scegli la variabile da prevedere (target) e le variabili "
        "esplicative. Il dataset viene diviso in **train** e **test**: "
        "i modelli imparano sul train e vengono valutati sul test, che "
        "non hanno mai visto. Valori mancanti e variabili categoriche "
        "sono gestiti automaticamente (calcolati solo sul train). "
        "Si usa il dataset pulito."
    ),
    *avviso_librerie,
    pn.Row(
        pn.Column(
            mod_target,
            mod_feature,
            mod_test_size,
            mod_soglia,
            mod_seed,
            mod_stratifica,
            mod_cv,
            mod_scelta,
            scheda_iperparametri,
            mod_allena,
            mod_esito,
            width=380
        ),
        pn.Column(
            area_modelli,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)


# ============================================================
# SCHEDA "PULIZIA DATI": LAYOUT
# ============================================================

tab_colonne = pn.Column(
    pn.pane.Markdown(
        "### Colonne\n"
        "Seleziona le colonne che ritieni non necessarie "
        "(puoi sceglierne più di una). La tabella ti aiuta a decidere: "
        "una colonna con un solo valore distinto o con molti "
        "mancanti è spesso poco utile."
    ),
    col_informazioni,
    pn.layout.Divider(),
    pn.Row(
        pn.Column(
            col_selezione,
            col_rimuovi,
            col_conferma,
            width=340
        ),
        pn.Column(
            col_esito,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)

tab_duplicati = pn.Column(
    pn.pane.Markdown(
        "### Righe duplicate\n"
        "Scegli le colonne da confrontare: due righe sono duplicate "
        "se hanno gli stessi valori in tutte le colonne selezionate."
    ),
    pn.Row(
        pn.Column(
            dup_colonne,
            dup_mantieni,
            pn.Row(dup_cerca, dup_rimuovi),
            dup_conferma,
            width=340
        ),
        pn.Column(
            dup_esito,
            dup_risultato,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)

tab_mancanti = pn.Column(
    pn.pane.Markdown("### Valori mancanti"),
    mis_statistiche,
    pn.layout.Divider(),
    pn.pane.Markdown(
        "Scegli le colonne, poi visualizza le righe interessate, "
        "rimuovile oppure imputa i valori mancanti."
    ),
    pn.Row(
        pn.Column(
            mis_colonne,
            pn.Row(mis_mostra, mis_rimuovi),
            pn.layout.Divider(),
            mis_metodo,
            mis_valore,
            mis_imputa,
            mis_conferma,
            width=340
        ),
        pn.Column(
            mis_esito,
            mis_risultato,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)

tab_outlier = pn.Column(
    pn.pane.Markdown(
        "### Outlier\n"
        "Scegli una variabile numerica e un metodo. "
        "Gli outlier sono mostrati in rosso."
    ),
    pn.Row(
        pn.Column(
            out_colonna,
            out_metodo,
            pn.Row(out_cerca, out_rimuovi),
            out_conferma,
            width=340
        ),
        pn.Column(
            out_esito,
            out_risultato,
            sizing_mode="stretch_width"
        ),
        sizing_mode="stretch_width"
    ),
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)

tab_dataset = pn.Column(
    pn.pane.Markdown(
        "### Dataset pulito\n"
        "Tutte le operazioni agiscono su una **copia** del dataset. "
        "L'originale resta in memoria e il file caricato non viene mai "
        "sovrascritto: il risultato si scarica come nuovo CSV."
    ),
    riepilogo_dataset,
    pn.pane.Markdown("#### Operazioni eseguite"),
    registro_operazioni,
    pn.layout.Divider(),
    pn.Row(
        selettore_separatore,
        pn.Column(
            pn.Spacer(height=20),
            scarica
        ),
        ripristina
    ),
    ripristino_conferma,
    ripristino_esito,
    styles=STILE_SCHEDA,
    sizing_mode="stretch_width"
)

sottoschede_pulizia = pn.Tabs(
    ("Colonne", tab_colonne),
    ("Duplicati", tab_duplicati),
    ("Valori mancanti", tab_mancanti),
    ("Outlier", tab_outlier),
    ("Dataset pulito", tab_dataset),
    dynamic=True,
    sizing_mode="stretch_width"
)

pulizia = pn.Column(
    barra_stato,
    sottoschede_pulizia,
    sizing_mode="stretch_width"
)

# Le due schede principali.
schede_principali = pn.Tabs(
    ("Esplorazione", esplorazione),
    ("Pulizia dati", pulizia),
    ("Modelli", scheda_modelli),
    dynamic=True,
    sizing_mode="stretch_width"
)


# ============================================================
# CARICAMENTO E AGGIORNAMENTO DEL DATASET
# ============================================================

def carica_dataset(event):

    if file_input.value is None:
        return

    try:

        # Legge il nuovo CSV.
        df = leggi_csv(
            file_input.value
        )

        if df.empty:

            messaggio.object = (
                "## Attenzione\n\n"
                "Il CSV è stato letto, ma non contiene righe di dati."
            )

            main.objects = [
                messaggio
            ]

            return

        # Il dataset originale non verrà mai modificato:
        # tutte le operazioni lavorano sulla copia "pulito".
        stato["originale"] = df
        stato["pulito"] = df.copy()
        stato["registro"] = []

        nome_file = file_input.filename or "dataset.csv"

        stato["nome_file"] = nome_file

        # Il file scaricato avrà un nome diverso dall'originale.
        scarica.filename = (
            os.path.splitext(nome_file)[0] + "_pulito.csv"
        )

        # Costruisce le due schede con il nuovo DataFrame.
        aggiorna_tutto()

        # Ricostruisce il contenuto principale.
        # Non cancella la barra laterale e il pulsante di caricamento.
        main.objects = [

            nome_dataset,

            schede_principali

        ]

        print(
            f"Dataset caricato correttamente: "
            f"{df.shape[0]} righe, {df.shape[1]} colonne."
        )

    except Exception as errore:

        import traceback

        traceback.print_exc()

        messaggio_errore = pn.pane.Markdown(
            f"""
            ## Errore durante il caricamento

            **Tipo di errore:** `{type(errore).__name__}`

            **Dettaglio:** `{str(errore)}`

            Controlla il formato del CSV oppure prova un altro file.
            """
        )

        main.objects = [
            messaggio_errore
        ]


# ============================================================
# SIDEBAR
# ============================================================

sidebar = pn.Column(

    pn.pane.Markdown(
        "# Data Explorer"
    ),

    pn.pane.Markdown(
        "Carica un CSV per visualizzare, analizzare e pulire i dati."
    ),

    file_input,

    pn.pane.Markdown(
        """
        ---
        ### Funzionalità

        - Panoramica del dataset
        - Statistiche descrittive
        - Distribuzioni numeriche
        - Frequenze categoriche
        - Confronti tra variabili
        - Sei grafici interattivi: scatter, linee, box plot,
          barre, heatmap e colori personalizzabili
        - Pulizia: colonne, duplicati, valori mancanti, outlier
        - Download del dataset pulito
        - Modelli di classificazione con split train/test
        """
    ),

    sizing_mode="stretch_width"

)


# ============================================================
# TEMPLATE PANEL
# ============================================================

dashboard = pn.template.FastListTemplate(

    title=TITOLO,

    sidebar=[
        sidebar
    ],

    main=[
        main
    ],

    accent_base_color="#2563eb",

    header_background="#1f2937"

)


# ============================================================
# EVENTO DI CARICAMENTO
# ============================================================

file_input.param.watch(
    carica_dataset,
    "value"
)


# ============================================================
# AVVIO
# ============================================================

dashboard.servable()
