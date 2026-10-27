"""Data Explorer — carica un dataset, esploralo e fai grafici.
Avvio:  pip install -r requirements.txt  &&  streamlit run app.py"""
import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Data Explorer", layout="wide")
ss = st.session_state


# ----------------------------------------------------------------- caricamento
@st.cache_data(show_spinner="Caricamento…")
def load(name: str, data: bytes) -> pd.DataFrame:
    ext = name.rsplit(".", 1)[-1].lower()
    buf = io.BytesIO(data)
    if ext in ("csv", "tsv", "txt"):
        try:
            return pd.read_csv(buf, sep=None, engine="python")
        except UnicodeDecodeError:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", encoding="latin-1")
    if ext in ("xlsx", "xls"):
        return pd.read_excel(buf)
    if ext == "json":
        try:
            return pd.read_json(buf)
        except ValueError:
            return pd.read_json(io.BytesIO(data), lines=True)
    if ext == "parquet":
        return pd.read_parquet(buf)
    raise ValueError(f"formato .{ext} non supportato")


st.sidebar.title("Data Explorer")
files = st.sidebar.file_uploader("Carica uno o più dataset", accept_multiple_files=True,
                                 type=["csv", "tsv", "txt", "xlsx", "xls", "json", "parquet"])
sets = {}
for f in files or []:
    try:
        sets[f.name] = load(f.name, f.getvalue())
    except Exception as e:
        st.sidebar.error(f"{f.name}: {e}")
if st.sidebar.checkbox("Usa un dataset di esempio (mance al ristorante)"):
    sets["esempio_tips"] = px.data.tips()
if not sets:
    st.title("Data Explorer")
    st.info("Carica un file dalla barra laterale (CSV, Excel, JSON, Parquet) per iniziare.")
    st.stop()

name = st.sidebar.selectbox("Dataset", list(sets)) if len(sets) > 1 else next(iter(sets))
raw = sets[name]

# ----------------------------------------------------------------- filtri (valgono per tutta l'app)
st.sidebar.subheader("Filtri")
df = raw
for c in st.sidebar.multiselect("Filtra per colonna", raw.columns.tolist()):
    s = raw[c]
    if pd.api.types.is_numeric_dtype(s) and s.notna().any():
        lo, hi = float(s.min()), float(s.max())
        if lo < hi:
            a, b = st.sidebar.slider(c, lo, hi, (lo, hi))
            df = df[df[c].between(a, b) | df[c].isna()]
    elif pd.api.types.is_datetime64_any_dtype(s):
        a, b = st.sidebar.date_input(c, (s.min(), s.max()))
        df = df[(df[c] >= pd.Timestamp(a)) & (df[c] <= pd.Timestamp(b))]
    elif s.nunique() <= 200:
        keep = st.sidebar.multiselect(c, sorted(s.dropna().astype(str).unique()))
        if keep:
            df = df[df[c].astype(str).isin(keep)]
    else:
        t = st.sidebar.text_input(f"{c} contiene")
        if t:
            df = df[df[c].astype(str).str.contains(t, case=False, na=False)]
sample_max = st.sidebar.number_input("Max punti nei grafici a dispersione", 1000, 1_000_000, 50_000, 1000)
st.sidebar.caption(f"{len(df):,} righe su {len(raw):,} · {df.shape[1]} colonne")

# ----------------------------------------------------------------- colori (scelti a runtime)
with st.sidebar.expander("🎨 Colori dei grafici"):
    PALETTES = ["Plotly", "D3", "G10", "T10", "Set2", "Pastel", "Bold", "Safe", "Vivid", "Dark24"]
    mode = st.radio("Colori per categorie e barre", ["Palette", "Personalizzati"], horizontal=True)
    if mode == "Palette":
        pal = st.selectbox("Palette", PALETTES)
        SEQ = getattr(px.colors.qualitative, pal)
    else:
        base = ["#1f6f5c", "#d9822b", "#3b6fb6", "#b8434a", "#7a5aa6", "#6b7770", "#c4a000", "#2a9d8f"]
        n = st.slider("Quanti colori", 1, 8, 3)
        SEQ = [st.color_picker(f"Colore {i + 1}" + (" (grafici semplici)" if i == 0 else ""), base[i], key=f"cp{i}")
               for i in range(n)]
    CONT = st.selectbox("Scala continua (densità, contingenza)", ["Viridis", "Plasma", "Cividis", "Turbo", "Blues", "Greens", "Reds"])
    DIV = st.selectbox("Scala per le correlazioni", ["RdBu_r", "RdYlGn", "PiYG", "Spectral_r", "PuOr"])
    if st.checkbox("Inverti scala continua"):
        CONT += "_r"
    TEMPLATE = st.selectbox("Sfondo / stile", ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation", "none"])
px.defaults.color_discrete_sequence = SEQ
px.defaults.color_continuous_scale = CONT

if df.empty:
    st.warning("I filtri non lasciano nessuna riga.")
    st.stop()

cols = df.columns.tolist()
num = df.select_dtypes("number").columns.tolist()
dt = df.select_dtypes("datetime").columns.tolist()
cat_low = [c for c in cols if c not in num and c not in dt and df[c].nunique() <= 20]
st.title(name)
t_tab, t_stat, t_one, t_two, t_multi = st.tabs(["Tabella", "Statistiche", "Grafico: una colonna",
                                                "Grafico: due colonne", "Panoramica"])


def show(fig, key):
    fig.update_layout(template=TEMPLATE)
    st.plotly_chart(fig, width="stretch", key=f"chart_{key}")
    a, b, _ = st.columns([1, 1, 4])
    a.download_button("Scarica HTML", fig.to_html(), f"{key}.html", key=f"h_{key}")
    pk = f"png_{key}_{hash(fig.to_json())}"
    if b.button("Prepara PNG", key=f"b_{key}"):
        try:
            ss[pk] = fig.to_image(format="png", scale=2)
        except Exception as e:
            st.warning(f"PNG non disponibile (serve kaleido e un browser Chrome): {e}")
    if ss.get(pk):
        st.download_button("Scarica PNG", ss[pk], f"{key}.png", key=f"d_{key}")


# ----------------------------------------------------------------- 1. tabella
with t_tab:
    a, b, c, d = st.columns([3, 2, 1, 1])
    shown = a.multiselect("Colonne da mostrare", cols, default=cols[:40])
    sort_by = b.multiselect("Ordina per (in ordine di priorità)", shown)
    asc = c.radio("Direzione", ["Crescente", "Decrescente"]) == "Crescente"
    nrows = d.number_input("Righe mostrate", 10, 100_000, 1000, 100)
    view = df[shown]
    if sort_by:
        view = view.sort_values(sort_by, ascending=asc)
    st.dataframe(view.head(int(nrows)), width="stretch", height=520)
    st.caption("Puoi anche ordinare cliccando sull'intestazione di una colonna, cercare con la lente in alto a destra "
               "della tabella e ingrandirla a schermo intero.")
    st.download_button("Scarica la vista come CSV", view.to_csv(index=False).encode(), f"{name}_vista.csv")

# ----------------------------------------------------------------- statistiche
with t_stat:
    m = st.columns(4)
    m[0].metric("Righe", f"{len(df):,}")
    m[1].metric("Colonne", df.shape[1])
    m[2].metric("Celle mancanti", f"{int(df.isna().sum().sum()):,}")
    m[3].metric("Righe duplicate", f"{int(df.duplicated().sum()):,}")

    st.subheader("Colonne")
    info = pd.DataFrame({"tipo": df.dtypes.astype(str), "mancanti": df.isna().sum(),
                         "% mancanti": (100 * df.isna().mean()).round(1), "valori unici": df.nunique()})
    st.dataframe(info, width="stretch")
    if num:
        st.subheader("Colonne numeriche")
        d = df[num].describe(percentiles=[.05, .25, .5, .75, .95]).T
        d["asimmetria"] = df[num].skew()
        d["curtosi"] = df[num].kurt()
        st.dataframe(d.round(3), width="stretch")
    other = [c for c in cols if c not in num]
    if other:
        st.subheader("Colonne non numeriche")
        st.dataframe(df[other].describe().T, width="stretch")

    st.subheader("Dettaglio di una colonna")
    c = st.selectbox("Colonna", cols, key="stat_col")
    s = df[c]
    left, right = st.columns(2)
    if pd.api.types.is_numeric_dtype(s):
        left.dataframe(s.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).to_frame(c).round(4), width="stretch")
        right.dataframe(s.value_counts().head(15).rename("frequenza").to_frame(), width="stretch")
    else:
        vc = s.value_counts(dropna=False).head(30)
        left.dataframe(pd.DataFrame({"frequenza": vc, "%": (100 * vc / len(s)).round(1)}), width="stretch")
    if len(num) > 1:
        st.subheader("Correlazioni")
        meth = st.radio("Metodo", ["pearson", "spearman", "kendall"], horizontal=True)
        C = df[num].corr(method=meth)
        st.dataframe(C.round(3).style.background_gradient(cmap=DIV, vmin=-1, vmax=1), width="stretch")

# ----------------------------------------------------------------- 2a. una colonna
with t_one:
    c = st.selectbox("Colonna", cols, key="one_col")
    s = df[c]
    color_opts = [None] + [x for x in cat_low if x != c]
    try:
        if pd.api.types.is_numeric_dtype(s):
            kind = st.radio("Tipo", ["Istogramma", "Box plot", "Violino", "ECDF (distribuzione cumulata)",
                                     "Valori in ordine di riga"], horizontal=True)
            a, b, d = st.columns(3)
            color = a.selectbox("Colora per", color_opts, key="one_color")
            log = b.checkbox("Scala logaritmica", key="one_log")
            if kind == "Istogramma":
                bins = d.slider("Intervalli", 5, 200, 30)
                norm = st.selectbox("Normalizzazione", ["conteggio", "percent", "probability density"])
                fig = px.histogram(df, x=c, color=color, nbins=bins, histnorm=None if norm == "conteggio" else norm,
                                   marginal="box", barmode="overlay", opacity=.75 if color else 1)
                if log: fig.update_yaxes(type="log")
            elif kind == "Box plot":
                fig = px.box(df, y=c, x=color, color=color, points="outliers")
                if log: fig.update_yaxes(type="log")
            elif kind == "Violino":
                fig = px.violin(df, y=c, x=color, color=color, box=True)
                if log: fig.update_yaxes(type="log")
            elif kind.startswith("ECDF"):
                fig = px.ecdf(df, x=c, color=color)
                if log: fig.update_xaxes(type="log")
            else:
                fig = px.line(df.reset_index(), x="index", y=c, color=color)
                if log: fig.update_yaxes(type="log")
        elif pd.api.types.is_datetime64_any_dtype(s):
            freq = st.selectbox("Raggruppa per", {"Giorno": "D", "Settimana": "W", "Mese": "MS", "Anno": "YS"})
            f = {"Giorno": "D", "Settimana": "W", "Mese": "MS", "Anno": "YS"}[freq]
            g = df.set_index(c).resample(f).size().rename("conteggio").reset_index()
            fig = px.line(g, x=c, y="conteggio", markers=True)
        else:
            kind = st.radio("Tipo", ["Barre (conteggi)", "Torta"], horizontal=True)
            top = st.slider("Mostra le prime N categorie", 3, 50, 15)
            vc = s.value_counts().head(top).rename_axis(c).reset_index(name="conteggio")
            fig = px.bar(vc, x=c, y="conteggio") if kind.startswith("Barre") else px.pie(vc, names=c, values="conteggio")
        show(fig, "una_colonna")
    except Exception as e:
        st.warning(f"Impossibile creare il grafico: {e}")

# ----------------------------------------------------------------- 2b. due colonne
with t_two:
    a, b = st.columns(2)
    x = a.selectbox("Asse X", cols, key="x_col")
    y = b.selectbox("Asse Y", [c for c in cols if c != x] or cols, key="y_col")
    nx, ny = x in num, y in num
    kinds = ["Dispersione", "Linea", "Barre (aggregazione di Y per X)", "Box plot", "Violino",
             "Densità 2D", "Tabella di contingenza"]
    kind = st.radio("Tipo", kinds, horizontal=True, index=0 if nx and ny else 2 if ny else 6 if not ny else 0)
    a, b, c3, d = st.columns(4)
    color = a.selectbox("Colora per", [None] + [c for c in cols if c not in (x, y) and (c in cat_low or c in num)], key="two_color")
    logx, logy = b.checkbox("X logaritmico"), c3.checkbox("Y logaritmico")
    try:
        if kind == "Dispersione":
            trend = d.selectbox("Linea di tendenza", [None, "ols", "lowess"])
            marg = st.checkbox("Mostra distribuzioni marginali")
            data = df if len(df) <= sample_max else df.sample(int(sample_max), random_state=0)
            if len(data) < len(df): st.caption(f"Mostrati {len(data):,} punti campionati su {len(df):,}.")
            fig = px.scatter(data, x=x, y=y, color=color, trendline=trend,
                             marginal_x="histogram" if marg else None, marginal_y="histogram" if marg else None, opacity=.7)
        elif kind == "Linea":
            fig = px.line(df.sort_values(x), x=x, y=y, color=color)
        elif kind.startswith("Barre"):
            agg = d.selectbox("Aggregazione", ["mean", "median", "sum", "count", "min", "max"])
            top = st.slider("Prime N categorie di X", 3, 100, 20)
            keys = [x] + ([color] if color else [])
            g = df.groupby(keys, dropna=False)[y].agg(agg).reset_index()
            g = g[g[x].isin(g.groupby(x)[y].sum().abs().nlargest(top).index)] if not nx else g
            fig = px.bar(g.sort_values(x), x=x, y=y, color=color, barmode="group", labels={y: f"{agg} di {y}"})
        elif kind == "Box plot":
            fig = px.box(df, x=x, y=y, color=color)
        elif kind == "Violino":
            fig = px.violin(df, x=x, y=y, color=color, box=True)
        elif kind == "Densità 2D":
            fig = px.density_heatmap(df, x=x, y=y, nbinsx=40, nbinsy=40, marginal_x="histogram", marginal_y="histogram")
        else:
            tx = df[x].isin(df[x].value_counts().head(30).index)
            ty = df[y].isin(df[y].value_counts().head(30).index)
            fig = px.imshow(pd.crosstab(df.loc[tx & ty, x], df.loc[tx & ty, y]), text_auto=True, aspect="auto",
                            labels={"color": "conteggio"})
        if logx: fig.update_xaxes(type="log")
        if logy: fig.update_yaxes(type="log")
        show(fig, "due_colonne")
    except Exception as e:
        st.warning(f"Impossibile creare il grafico con queste colonne: {e}")

# ----------------------------------------------------------------- panoramica
with t_multi:
    what = st.radio("Grafico", ["Matrice di correlazione", "Scatter matrix", "Valori mancanti"], horizontal=True)
    try:
        if what == "Matrice di correlazione":
            meth = st.radio("Metodo", ["pearson", "spearman", "kendall"], horizontal=True, key="m_meth")
            fig = px.imshow(df[num].corr(method=meth), text_auto=".2f" if len(num) <= 15 else False,
                            zmin=-1, zmax=1, color_continuous_scale=DIV, aspect="auto")
        elif what == "Scatter matrix":
            pick = st.multiselect("Colonne (max 8)", num, default=num[:4])[:8]
            col = st.selectbox("Colora per", [None] + cat_low, key="sm_col")
            data = df if len(df) <= 5000 else df.sample(5000, random_state=0)
            fig = px.scatter_matrix(data, dimensions=pick, color=col, opacity=.5)
            fig.update_traces(diagonal_visible=False, marker_size=3)
            fig.update_layout(height=700)
        else:
            mis = (100 * df.isna().mean()).sort_values(ascending=False)
            fig = px.bar(mis[mis > 0], labels={"value": "% mancanti", "index": ""}, title="Percentuale di valori mancanti")
            if mis.max() == 0: st.success("Nessun valore mancante.")
        show(fig, "panoramica")
    except Exception as e:
        st.warning(f"Impossibile creare il grafico: {e}")
