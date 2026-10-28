
import panel as pn
import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go


pn.extension("plotly")


# ============================================================
# CREAZIONE DEL GRAFICO
# ============================================================

def crea_grafico(df, colonna, gruppo=None):

    fig = go.Figure()

    colonne_numeriche = df.select_dtypes(
        include="number"
    ).columns

    # ========================================================
    # VARIABILE NUMERICA
    # ========================================================

    if colonna in colonne_numeriche:

        if gruppo is None:

            dati = df[colonna].dropna()

            fig.add_trace(
                go.Histogram(
                    x=dati,
                    nbinsx=20,
                    name=colonna,
                    marker=dict(
                        line=dict(
                            color="black",
                            width=0.5
                        )
                    ),
                    hovertemplate=(
                        f"{colonna}: %{{x}}<br>"
                        "Frequenza: %{y}"
                        "<extra></extra>"
                    )
                )
            )

            fig.update_layout(
                yaxis_title="Frequenza"
            )

        else:

            dati_validi = df[
                [colonna, gruppo]
            ].dropna(
                subset=[colonna, gruppo]
            )

            gruppi = dati_validi[gruppo].unique()

            if len(dati_validi) > 0 and len(gruppi) > 0:

                # Bin condivisi tra tutti i gruppi
                bins = np.histogram_bin_edges(
                    dati_validi[colonna],
                    bins=20
                )

                centri_bin = (
                    bins[:-1] + bins[1:]
                ) / 2

                larghezza_bin = bins[1] - bins[0]

                numero_gruppi = len(gruppi)

                larghezza_totale = larghezza_bin * 0.95

                larghezza_barra = (
                    larghezza_totale / numero_gruppi
                )

                # Frequenze assolute per ciascun gruppo
                frequenze_gruppi = {}

                for valore in gruppi:

                    dati_gruppo = dati_validi.loc[
                        dati_validi[gruppo] == valore,
                        colonna
                    ]

                    frequenze, _ = np.histogram(
                        dati_gruppo,
                        bins=bins
                    )

                    frequenze_gruppi[valore] = frequenze

                # Totale di osservazioni per ciascun bin
                totali_bin = np.zeros(
                    len(centri_bin),
                    dtype=float
                )

                for frequenze in frequenze_gruppi.values():
                    totali_bin += frequenze

                # Percentuale di ciascun gruppo nel singolo bin
                for valore in gruppi:

                    frequenze = frequenze_gruppi[valore]

                    percentuali = np.divide(
                        frequenze * 100,
                        totali_bin,
                        out=np.zeros(
                            len(frequenze),
                            dtype=float
                        ),
                        where=totali_bin != 0
                    )

                    fig.add_trace(
                        go.Bar(
                            x=centri_bin,
                            y=percentuali,
                            width=larghezza_barra * 0.95,
                            name=str(valore),
                            customdata=frequenze,
                            hovertemplate=(
                                f"{colonna}: %{{x:.2f}}<br>"
                                f"{gruppo}: {valore}<br>"
                                "Percentuale nel bin: %{y:.2f}%<br>"
                                "Osservazioni: %{customdata}"
                                "<extra></extra>"
                            )
                        )
                    )

                fig.update_layout(
                    barmode="group",
                    yaxis_title="Percentuale nel bin (%)",
                    legend_title_text=gruppo
                )

                fig.update_yaxes(
                    range=[0, 100]
                )

            else:

                fig.add_annotation(
                    text="Nessun dato disponibile",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False
                )

        fig.update_layout(
            title=f"Distribuzione di {colonna}",
            xaxis_title=colonna
        )

    # ========================================================
    # VARIABILE CATEGORIALE
    # ========================================================

    else:

        dati = df[[colonna]].copy()

        dati[colonna] = (
            dati[colonna]
            .fillna("Mancante")
            .astype(str)
        )

        if gruppo is None:

            frequenze = dati[colonna].value_counts()

            fig.add_trace(
                go.Bar(
                    x=frequenze.index,
                    y=frequenze.values,
                    name=colonna,
                    marker=dict(
                        line=dict(
                            color="black",
                            width=0.5
                        )
                    ),
                    hovertemplate=(
                        f"{colonna}: %{{x}}<br>"
                        "Frequenza: %{y}"
                        "<extra></extra>"
                    )
                )
            )

        else:

            dati = df[
                [colonna, gruppo]
            ].copy()

            dati[colonna] = (
                dati[colonna]
                .fillna("Mancante")
                .astype(str)
            )

            dati[gruppo] = (
                dati[gruppo]
                .fillna("Mancante")
                .astype(str)
            )

            frequenze = pd.crosstab(
                dati[colonna],
                dati[gruppo]
            )

            for valore in frequenze.columns:

                fig.add_trace(
                    go.Bar(
                        x=frequenze.index,
                        y=frequenze[valore].values,
                        name=str(valore),
                        hovertemplate=(
                            f"{colonna}: %{{x}}<br>"
                            f"{gruppo}: {valore}<br>"
                            "Frequenza: %{y}"
                            "<extra></extra>"
                        )
                    )
                )

            fig.update_layout(
                barmode="group",
                legend_title_text=gruppo
            )

        fig.update_layout(
            title=f"Frequenze di {colonna}",
            xaxis_title=colonna,
            yaxis_title="Frequenza"
        )

        fig.update_xaxes(
            tickangle=-45
        )

    # ========================================================
    # ASPETTO E INTERATTIVITÀ
    # ========================================================

    fig.update_layout(
        autosize=True,
        margin=dict(
            l=45,
            r=15,
            t=55,
            b=65
        ),
        font=dict(
            size=11
        ),
        hovermode="closest",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0
        )
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(128,128,128,0.2)"
    )

    return fig


# ============================================================
# STATISTICHE DEL DATASET
# ============================================================

def crea_statistiche(df):

    numero_righe = len(df)
    numero_colonne = len(df.columns)

    # Celle mancanti
    totale_celle = df.size

    valori_mancanti = int(
        df.isna().sum().sum()
    )

    if totale_celle > 0:

        percentuale_mancanti = (
            valori_mancanti / totale_celle
        ) * 100

    else:

        percentuale_mancanti = 0

    # Righe con almeno un valore mancante
    righe_incomplete = int(
        df.isna().any(axis=1).sum()
    )

    # Righe duplicate
    righe_duplicate = int(
        df.duplicated().sum()
    )

    # Tipi di variabili
    colonne_numeriche = len(
        df.select_dtypes(
            include="number"
        ).columns
    )

    colonne_categoriali = (
        numero_colonne - colonne_numeriche
    )

    statistiche = [
        ("Righe", f"{numero_righe:,}"),
        ("Colonne", f"{numero_colonne:,}"),
        ("Valori mancanti", f"{valori_mancanti:,}"),
        ("Missing (%)", f"{percentuale_mancanti:.2f}%"),
        ("Righe incomplete", f"{righe_incomplete:,}"),
        ("Duplicati", f"{righe_duplicate:,}"),
        ("Numeriche", f"{colonne_numeriche:,}"),
        ("Categoriali", f"{colonne_categoriali:,}")
    ]

    schede = []

    for titolo, valore in statistiche:

        scheda = pn.Card(
            pn.pane.Markdown(
                f"<div style='text-align:center;"
                f"font-size:22px;font-weight:bold;'>"
                f"{valore}</div>"
            ),
            title=titolo,
            height=90,
            sizing_mode="stretch_width",
            margin=3
        )

        schede.append(scheda)

    pannello_statistiche = pn.GridBox(
        *schede,
        ncols=4,
        sizing_mode="stretch_width"
    )

    return pannello_statistiche


# ============================================================
# CREAZIONE DEI SEI GRAFICI
# ============================================================

def crea_istogrammi(df):

    grafici = []

    tutte_colonne = list(df.columns)

    colonne_numeriche = list(
        df.select_dtypes(
            include="number"
        ).columns
    )

    colonne_categoriali = [
        colonna
        for colonna in tutte_colonne
        if colonna not in colonne_numeriche
    ]

    # Prima le variabili numeriche
    colonne = (
        colonne_numeriche + colonne_categoriali
    )[:6]

    # Variabili utilizzabili per la suddivisione
    colonne_gruppo = []

    for colonna in tutte_colonne:

        numero_categorie = df[colonna].nunique(
            dropna=True
        )

        if numero_categorie <= 3:
            colonne_gruppo.append(colonna)

    opzioni_gruppo = [
        "Nessuna"
    ] + colonne_gruppo

    for colonna in colonne:

        # ----------------------------------------------------
        # SELETTORE DELLA VARIABILE
        # ----------------------------------------------------

        selettore_variabile = pn.widgets.Select(
            name="Variabile",
            options=tutte_colonne,
            value=colonna,
            sizing_mode="stretch_width"
        )

        # ----------------------------------------------------
        # SUDDIVISIONE PREDEFINITA
        # ----------------------------------------------------

        if (
            "Depression" in opzioni_gruppo
            and colonna != "Depression"
        ):

            gruppo_iniziale = "Depression"

        else:

            gruppo_iniziale = "Nessuna"

        opzioni_iniziali = [
            valore
            for valore in opzioni_gruppo
            if (
                valore == "Nessuna"
                or valore != colonna
            )
        ]

        selettore_gruppo = pn.widgets.Select(
            name="Suddividi per",
            options=opzioni_iniziali,
            value=gruppo_iniziale,
            sizing_mode="stretch_width"
        )

        # ----------------------------------------------------
        # GRAFICO INIZIALE
        # ----------------------------------------------------

        gruppo_grafico = (
            None
            if gruppo_iniziale == "Nessuna"
            else gruppo_iniziale
        )

        fig = crea_grafico(
            df,
            colonna,
            gruppo_grafico
        )

        pannello_grafico = pn.pane.Plotly(
            fig,
            sizing_mode="stretch_width",
            height=260,
            config={
                "responsive": True,
                "displayModeBar": True,
                "scrollZoom": True
            }
        )

        # ----------------------------------------------------
        # AGGIORNAMENTO DEL GRAFICO
        # ----------------------------------------------------

        def aggiorna_grafico(
            event=None,
            variabile=selettore_variabile,
            gruppo=selettore_gruppo,
            pannello=pannello_grafico
        ):

            colonna_selezionata = variabile.value

            # Aggiorna le opzioni dei gruppi
            opzioni_nuove = [
                valore
                for valore in opzioni_gruppo
                if (
                    valore == "Nessuna"
                    or valore != colonna_selezionata
                )
            ]

            if gruppo.value not in opzioni_nuove:

                gruppo.value = "Nessuna"

            gruppo.options = opzioni_nuove

            gruppo_selezionato = gruppo.value

            if gruppo_selezionato == "Nessuna":

                gruppo_selezionato = None

            nuova_figura = crea_grafico(
                df,
                colonna_selezionata,
                gruppo_selezionato
            )

            pannello.object = nuova_figura

        selettore_variabile.param.watch(
            aggiorna_grafico,
            "value"
        )

        selettore_gruppo.param.watch(
            aggiorna_grafico,
            "value"
        )

        # ----------------------------------------------------
        # SCHEDA DEL GRAFICO
        # ----------------------------------------------------

        card = pn.Card(
            pn.Column(
                pn.Row(
                    selettore_variabile,
                    selettore_gruppo,
                    sizing_mode="stretch_width"
                ),
                pannello_grafico,
                sizing_mode="stretch_width"
            ),
            title=colonna,
            height=340,
            sizing_mode="stretch_width",
            margin=4
        )

        # Aggiorna il titolo della scheda
        def aggiorna_titolo(
            event,
            card_riferimento=card,
            variabile=selettore_variabile
        ):

            card_riferimento.title = variabile.value

        selettore_variabile.param.watch(
            aggiorna_titolo,
            "value"
        )

        grafici.append(card)

    return grafici


# ============================================================
# UPLOAD CSV
# ============================================================

file_input = pn.widgets.FileInput(
    name="Carica CSV",
    accept=".csv"
)


# ============================================================
# SIDEBAR
# ============================================================

sidebar = pn.Column(
    "# Data Explorer",
    file_input,
    width=250
)


# ============================================================
# CONTENITORE DEI GRAFICI
# ============================================================

area_plot = pn.GridBox(
    ncols=3,
    sizing_mode="stretch_width"
)


# ============================================================
# MESSAGGIO INIZIALE
# ============================================================

messaggio_iniziale = pn.pane.Markdown(
    "## Benvenuto in Data Explorer\n\n"
    "Carica un file CSV dalla barra laterale per iniziare."
)


# ============================================================
# CONTENITORE PRINCIPALE
# ============================================================

main = pn.Column(
    messaggio_iniziale,
    sizing_mode="stretch_width"
)


# ============================================================
# TEMPLATE
# ============================================================

dashboard = pn.template.FastListTemplate(
    title="Data Explorer",
    sidebar=[sidebar],
    main=[main]
)


# ============================================================
# CARICAMENTO DEL DATASET
# ============================================================

def carica_dataset(event):

    if file_input.value is None:
        return

    try:

        df = pd.read_csv(
            io.BytesIO(file_input.value),
            sep=None,
            engine="python"
        )

        if df.empty or len(df.columns) == 0:

            main.objects = [
                pn.pane.Markdown(
                    "Il dataset è vuoto."
                )
            ]

            return

        # Statistiche del dataset
        pannello_statistiche = crea_statistiche(df)

        # Grafici
        grafici = crea_istogrammi(df)

        # Configura la griglia
        area_plot.clear()

        area_plot.ncols = 3

        area_plot.sizing_mode = "stretch_width"

        for grafico in grafici:
            area_plot.append(grafico)

        # Contenitore complessivo
        contenuto = pn.Column(
            pn.pane.Markdown(
                "## Panoramica del dataset"
            ),
            pannello_statistiche,
            pn.layout.Divider(),
            pn.pane.Markdown(
                "## Esplorazione delle variabili"
            ),
            area_plot,
            sizing_mode="stretch_width"
        )

        main.objects = [
            contenuto
        ]

        # Svuota la sidebar
        dashboard.sidebar.clear()

    except Exception as errore:

        import traceback

        traceback.print_exc()

        main.objects = [
            pn.pane.Markdown(
                f"**Errore durante il caricamento:** {errore}"
            )
        ]


# ============================================================
# EVENTI
# ============================================================

file_input.param.watch(
    carica_dataset,
    "value"
)


# ============================================================
# AVVIO
# ============================================================

dashboard.servable()
