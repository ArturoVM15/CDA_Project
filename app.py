"""
Dashboard interactivo — Proyecto Integrador: Cambio Climático y Calentamiento Global
Relación entre emisiones de CO2, consumo de combustibles fósiles y temperatura.

Ejecutar localmente:   streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score

# ----------------------------------------------------------------------
# Configuración general
# ----------------------------------------------------------------------
st.set_page_config(page_title="Dashboard Clima · CO₂ y Temperatura",
                  layout="wide")

COL_CO2 = "Emision_anual_de_CO2"          # toneladas
COL_FOS = "Consumo_de_combustibles_fosiles"  # TWh
COL_TMP = "Average surface temperature"      # °C

PALETTE = px.colors.qualitative.Set2


# ----------------------------------------------------------------------
# Carga y preparación de datos (cacheado)
# ----------------------------------------------------------------------
@st.cache_data
def cargar_datos(path="Base_de_Datos_Proyecto.csv"):
    df = pd.read_csv(path, decimal=",")
    for c in [COL_CO2, COL_FOS, COL_TMP]:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False),
                              errors="coerce")
    df = df.dropna(subset=[COL_CO2, COL_FOS, COL_TMP]).copy()

    # Anomalía por país: temperatura menos su línea base 1980-2000 (quita la geografía)
    base = (df[df["Anio"].between(1980, 2000)]
            .groupby("Pais")[COL_TMP].mean())
    df["anom"] = df[COL_TMP] - df["Pais"].map(base)
    return df


@st.cache_data
def serie_global(df):
    """Agregado mundial por año."""
    return (df.groupby("Anio")
              .agg(co2=(COL_CO2, "sum"),
                   fosil=(COL_FOS, "sum"),
                   temp_anom=("anom", "mean"))
              .reset_index().dropna())


@st.cache_data
def correlaciones_por_pais(df):
    filas = []
    for pais, d in df.groupby("Pais"):
        filas.append({
            "Pais": pais,
            "corr_CO2": d[COL_CO2].corr(d[COL_TMP]),
            "corr_fosil": d[COL_FOS].corr(d[COL_TMP]),
        })
    return pd.DataFrame(filas)


@st.cache_data
def cargar_kaggle(path="Kaggle_Climate_Dataset.csv"):
    """Dataset sintético inicial (exploratorio). Devuelve None si no está."""
    try:
        k = pd.read_csv(path)
        return k if len(k) and "predicted_temperature_2050" in k.columns else None
    except Exception:
        return None


df = cargar_datos()
g = serie_global(df)
kaggle = cargar_kaggle()

# Diccionario para trabajar CO2 / Fósil con la misma interfaz
VARS = {
    "CO₂ (emisiones, t)": {"col": COL_CO2, "gcol": "co2", "unidad": "toneladas",
                           "escala": 1e9, "esc_lbl": "Gt"},
    "Consumo fósil (TWh)": {"col": COL_FOS, "gcol": "fosil", "unidad": "TWh",
                            "escala": 1000, "esc_lbl": "miles de TWh"},
}

# ----------------------------------------------------------------------
# Barra lateral — navegación
# ----------------------------------------------------------------------
st.sidebar.title("Navegación")
seccion = st.sidebar.radio(
    "Ir a:",
    ["Inicio", "Exploración (EDA)", "Relación global",
     "Relación por país", "Tendencias",
     "Proyección a 2050", "Comparación de modelos",
     "Exploración inicial (Kaggle)", "Conclusiones"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Datos: {len(df)} registros · "
                   f"{df['Pais'].nunique()} países · "
                   f"{int(df['Anio'].min())}–{int(df['Anio'].max())}")


# ======================================================================
# 1. INICIO
# ======================================================================
if seccion == "Inicio":
    st.title("Cambio Climático y Calentamiento Global")
    st.markdown("### Relación entre emisiones de CO₂, consumo de combustibles "
                "fósiles y la temperatura media")

    st.markdown(
        "Este dashboard analiza la relación entre las **emisiones de CO₂**, el "
        "**consumo de combustibles fósiles** y el **incremento de la temperatura "
        "media**, con datos de 10 países entre 1970 y 2024, para identificar "
        "tendencias y proyectar el calentamiento futuro."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Países", df["Pais"].nunique())
    c2.metric("Años", f"{int(df['Anio'].min())}–{int(df['Anio'].max())}")
    corr_glob = g["co2"].corr(g["temp_anom"])
    c3.metric("Correlación CO₂–Temp (global)", f"{corr_glob:.2f}")
    anom_ult = g.loc[g["Anio"].idxmax(), "temp_anom"]
    c4.metric("Anomalía último año", f"+{anom_ult:.2f} °C")

    st.markdown("---")
    st.subheader("Temperatura media global (anomalía) a lo largo del tiempo")
    fig = px.line(g, x="Anio", y="temp_anom", markers=True,
                  labels={"Anio": "Año", "temp_anom": "Anomalía de temperatura (°C)"})
    fig.update_traces(line_color="#c0392b")
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig, use_container_width=True)

    st.info("Usá el menú de la izquierda para recorrer el análisis: exploración, "
            "relación (global y por país), proyección a 2050 y comparación de modelos.")


# ======================================================================
# 2. EXPLORACIÓN (EDA)
# ======================================================================
elif seccion == "Exploración (EDA)":
    st.title("Análisis exploratorio")

    paises = st.multiselect("Filtrar países",
                            sorted(df["Pais"].unique()),
                            default=sorted(df["Pais"].unique()))
    d = df[df["Pais"].isin(paises)] if paises else df

    variable = st.selectbox("Variable a visualizar",
                            [COL_TMP, COL_CO2, COL_FOS])
    st.subheader(f"Evolución temporal · {variable}")
    fig = px.line(d.sort_values("Anio"), x="Anio", y=variable, color="Pais",
                  color_discrete_sequence=PALETTE,
                  labels={"Anio": "Año"})
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Estadística descriptiva")
        st.dataframe(d[[COL_CO2, COL_FOS, COL_TMP]].describe().round(2),
                     use_container_width=True)
    with col2:
        st.subheader("Temperatura media por país")
        tmp_pais = (d.groupby("Pais")[COL_TMP].mean()
                    .sort_values().round(1).reset_index())
        st.dataframe(tmp_pais, use_container_width=True, hide_index=True)

    with st.expander("Ver datos crudos"):
        st.dataframe(d, use_container_width=True)

    st.subheader("Distribución de las variables (boxplots)")
    cols = st.columns(3)
    for ax_col, (col, titulo) in zip(
            cols, [(COL_CO2, "Emisiones de CO₂ (t)"),
                   (COL_FOS, "Consumo fósil (TWh)"),
                   (COL_TMP, "Temperatura (°C)")]):
        figb = px.box(d, y=col, points="outliers", title=titulo,
                      color_discrete_sequence=["#2e6b9a"])
        figb.update_layout(height=350, showlegend=False, yaxis_title="")
        ax_col.plotly_chart(figb, use_container_width=True)


# ======================================================================
# 3. RELACIÓN GLOBAL
# ======================================================================
elif seccion == "Relación global":
    st.title("Relación a nivel global")
    st.markdown("Sumamos todas las emisiones/consumos del mundo por año y las "
                "cruzamos con la anomalía de temperatura global. **Es donde la "
                "relación aparece con claridad.**")

    etiqueta = st.radio("Variable predictora", list(VARS.keys()), horizontal=True)
    v = VARS[etiqueta]
    gcol, escala, esc_lbl = v["gcol"], v["escala"], v["esc_lbl"]

    X = g[[gcol]].values
    y = g["temp_anom"].values
    modelo = LinearRegression().fit(X, y)
    r2 = modelo.score(X, y)
    corr = g[gcol].corr(g["temp_anom"])
    cv = cross_val_score(modelo, X, y,
                         cv=KFold(5, shuffle=True, random_state=1),
                         scoring="r2").mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Correlación", f"{corr:.2f}")
    c2.metric("R² (in-sample)", f"{r2:.2f}")
    c3.metric("R² (validación cruzada)", f"{cv:.2f}")

    xs = np.linspace(g[gcol].min(), g[gcol].max(), 100)
    ys = modelo.predict(xs.reshape(-1, 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g[gcol] / escala, y=g["temp_anom"], mode="markers",
                             marker=dict(size=9, color=g["Anio"],
                                         colorscale="Viridis", showscale=True,
                                         colorbar=dict(title="Año")),
                             text=g["Anio"], name="Observado"))
    fig.add_trace(go.Scatter(x=xs / escala, y=ys, mode="lines",
                             line=dict(color="#c0392b", width=3),
                             name=f"Ajuste (R²={r2:.2f})"))
    fig.update_layout(xaxis_title=f"{etiqueta} — global ({esc_lbl})",
                      yaxis_title="Anomalía de temperatura (°C)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Por cada unidad extra ({esc_lbl}), la anomalía sube "
               f"**{modelo.coef_[0]*escala:+.3f} °C**.")
    st.caption("Nota: CO₂ y consumo fósil están correlacionados al 0.99 — son "
               "prácticamente la misma variable. Por eso los dos modelos dan casi "
               "idéntico y NO deben combinarse en una sola regresión.")


# ======================================================================
# 4. RELACIÓN POR PAÍS
# ======================================================================
elif seccion == "Relación por país":
    st.title("Relación por país")
    st.markdown("¿La temperatura de cada país sigue a **sus propias** emisiones? "
                "Acá aparece el hallazgo más interesante del proyecto.")

    etiqueta = st.radio("Variable", list(VARS.keys()), horizontal=True)
    v = VARS[etiqueta]
    col = v["col"]
    corr_col = "corr_CO2" if col == COL_CO2 else "corr_fosil"

    tabla = correlaciones_por_pais(df).sort_values(corr_col, ascending=False)
    tabla["Signo"] = np.where(tabla[corr_col] > 0, "Positiva", "Negativa")

    fig = px.bar(tabla.sort_values(corr_col), x=corr_col, y="Pais",
                 orientation="h", color="Signo",
                 color_discrete_map={"Positiva": "#c0392b", "Negativa": "#2471a3"},
                 text=tabla.sort_values(corr_col)[corr_col].round(2),
                 labels={corr_col: "Correlación con la temperatura"})
    fig.add_vline(x=0, line_color="black")
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.warning("**Francia, Alemania y Reino Unido dan correlación NEGATIVA**: "
               "redujeron sus emisiones durante décadas y aun así se calentaron. "
               "La temperatura de un país no responde a sus propias emisiones, "
               "sino al CO₂ **global**.")

    st.subheader("Explorar un país")
    pais_sel = st.selectbox("País", tabla["Pais"].tolist())
    d = df[df["Pais"] == pais_sel].sort_values("Anio")
    r = d[col].corr(d[COL_TMP])
    color = "#c0392b" if r > 0 else "#2471a3"
    fig2 = px.scatter(d, x=col, y=COL_TMP, color="Anio",
                      color_continuous_scale="Viridis",
                      labels={col: etiqueta, COL_TMP: "Temperatura (°C)"},
                      title=f"{pais_sel}  ·  correlación = {r:+.2f}")
    # Línea de tendencia calculada a mano (evita depender de statsmodels)
    coef = np.polyfit(d[col], d[COL_TMP], 1)
    xs = np.linspace(d[col].min(), d[col].max(), 50)
    fig2.add_trace(go.Scatter(x=xs, y=np.polyval(coef, xs), mode="lines",
                              line=dict(color=color, width=3),
                              name="Tendencia"))
    st.plotly_chart(fig2, use_container_width=True)


# ======================================================================
# 5. PROYECCIÓN A 2050
# ======================================================================
elif seccion == "Proyección a 2050":
    st.title("Proyección de calentamiento a 2050")

    modo = st.radio("Escala", ["🌍 Global", "🗺️ Por país"], horizontal=True)

    # ---------------- Proyección GLOBAL ----------------
    if modo == "🌍 Global":
        st.markdown("Encadenamos dos modelos: **año → variable** (tendencia) y luego "
                    "**variable → temperatura** (modelo de relación). No predecimos la "
                    "temperatura desde el año directamente.")

        etiqueta = st.radio("Driver de la proyección", list(VARS.keys()), horizontal=True)
        v = VARS[etiqueta]
        gcol = v["gcol"]

        modelo_rel = LinearRegression().fit(g[[gcol]], g["temp_anom"])
        m_trend = LinearRegression().fit(g[["Anio"]], g[gcol])

        # Backtest honesto
        tr = g[g["Anio"] <= 2012]
        te = g[g["Anio"] > 2012]
        m_bt = LinearRegression().fit(tr[["Anio"]], tr[gcol])
        r2_bt = r2_score(te[gcol], m_bt.predict(te[["Anio"]]))

        anios_fut = pd.DataFrame({"Anio": np.arange(2025, 2051)})
        var_fut = pd.DataFrame({gcol: m_trend.predict(anios_fut)})
        temp_fut = modelo_rel.predict(var_fut)

        c1, c2 = st.columns(2)
        c1.metric("Anomalía proyectada 2050", f"+{temp_fut[-1]:.2f} °C")
        c2.metric("R² backtest tendencia (2013–24)", f"{r2_bt:.2f}",
                  help="Negativo = la tendencia lineal no es confiable. "
                       "Tratar como ESCENARIO, no como predicción exacta.")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=g["Anio"], y=g["temp_anom"], mode="markers",
                                 marker=dict(color="gray", size=6), name="Observado"))
        fig.add_trace(go.Scatter(x=g["Anio"], y=modelo_rel.predict(g[[gcol]]),
                                 mode="lines", line=dict(color="#2471a3", width=3),
                                 name="Modelo de relación (ajuste)"))
        fig.add_trace(go.Scatter(x=anios_fut["Anio"], y=temp_fut, mode="lines",
                                 line=dict(color="#c0392b", width=3, dash="dash"),
                                 name="Proyección 2025–2050"))
        fig.add_vline(x=2024, line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Año",
                          yaxis_title="Anomalía de temperatura (°C)",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        if r2_bt < 0:
            st.warning("El backtest da R² negativo: las emisiones globales no siguen "
                       "una recta perfecta (se aplanaron tras ~2012). Presentar como "
                       "**escenario de tendencia lineal**, no como predicción dura.")

        st.markdown("---")
        if st.checkbox("Mostrar modelos baseline de temperatura (naive, media móvil, tendencia)"):
            gt = df.groupby("Anio")[COL_TMP].mean()
            m_tr = LinearRegression().fit(gt.index.values.reshape(-1, 1), gt.values)
            fut_b = np.arange(int(gt.index.max()) + 1, 2051)
            naive = gt.iloc[-1]
            ma10 = gt.iloc[-10:].mean()
            figb = go.Figure()
            figb.add_trace(go.Scatter(x=gt.index, y=gt.values, mode="lines",
                                      line=dict(color="black"), name="Histórico real"))
            figb.add_trace(go.Scatter(x=fut_b, y=[naive]*len(fut_b), mode="lines",
                                      line=dict(color="#c0392b", dash="dash"),
                                      name=f"Naive ({naive:.2f} °C)"))
            figb.add_trace(go.Scatter(x=fut_b, y=[ma10]*len(fut_b), mode="lines",
                                      line=dict(color="#2e8b57", dash="dash"),
                                      name=f"Media móvil 10y ({ma10:.2f} °C)"))
            figb.add_trace(go.Scatter(x=fut_b, y=m_tr.predict(fut_b.reshape(-1, 1)),
                                      mode="lines", line=dict(color="#2471a3", width=3),
                                      name="Tendencia lineal"))
            figb.add_vline(x=int(gt.index.max()), line_dash="dot", line_color="gray")
            figb.update_layout(xaxis_title="Año", yaxis_title="Temperatura global (°C)",
                               title="Modelos baseline vs tendencia",
                               legend=dict(orientation="h", y=1.1))
            st.plotly_chart(figb, use_container_width=True)
            st.caption("Los baseline (naive y media móvil) son líneas planas: sirven "
                       "como referencia mínima. La tendencia lineal es el modelo que "
                       "captura el calentamiento.")

    # ---------------- Proyección POR PAÍS ----------------
    else:
        st.markdown("Proyección de cada variable a 2050 por país, con **regresión "
                    "lineal** (el modelo elegido). La línea punteada es la extrapolación.")

        c1, c2 = st.columns(2)
        pais = c1.selectbox("País", sorted(df["Pais"].unique()))
        col_map = {"Temperatura (°C)": COL_TMP,
                   "Emisiones de CO₂ (t)": COL_CO2,
                   "Consumo fósil (TWh)": COL_FOS}
        etiqueta = c2.selectbox("Variable", list(col_map.keys()))
        col = col_map[etiqueta]

        d = df[df["Pais"] == pais].sort_values("Anio")
        modelo = LinearRegression().fit(d[["Anio"]], d[col])
        fut = pd.DataFrame({"Anio": np.arange(int(d["Anio"].max()) + 1, 2051)})
        pred = modelo.predict(fut)

        ver_svr = st.checkbox("Comparar con proyección SVR",
                              help="El SVR (kernel RBF) no extrapola: tiende a "
                                   "'aplanarse' o revertir a la media fuera del rango "
                                   "de años entrenado. Por eso se eligió la lineal.")

        st.metric(f"{etiqueta} proyectada a 2050 (Lineal)",
                  f"{modelo.predict(pd.DataFrame({'Anio':[2050]}))[0]:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["Anio"], y=d[col], mode="lines+markers",
                                 line=dict(color="#2471a3"), name="Histórico"))
        fig.add_trace(go.Scatter(x=fut["Anio"], y=pred, mode="lines",
                                 line=dict(color="#c0392b", dash="dash"),
                                 name="Proyección Lineal"))
        if ver_svr:
            svr = make_pipeline(StandardScaler(),
                                SVR(kernel="rbf", C=10, gamma="scale")).fit(d[["Anio"]], d[col])
            pred_svr = svr.predict(fut)
            fig.add_trace(go.Scatter(x=fut["Anio"], y=pred_svr, mode="lines",
                                     line=dict(color="#2e8b57", dash="dot", width=3),
                                     name="Proyección SVR"))
        fig.add_vline(x=int(d["Anio"].max()), line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Año", yaxis_title=etiqueta,
                          title=f"{pais} · proyección a 2050",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

        if ver_svr:
            st.info("Fijate cómo la línea verde (SVR) se aplana o incluso baja al "
                    "extrapolar — a veces predice enfriamiento. Ese es el motivo por "
                    "el que el proyecto se quedó con la regresión lineal.")


# ======================================================================
# 4b. TENDENCIAS
# ======================================================================
elif seccion == "Tendencias":
    st.title("Tendencias temporales")
    st.markdown("Evolución de temperatura, emisiones de CO₂ y consumo de combustibles "
                "fósiles, con su línea de tendencia lineal.")

    modo = st.radio("Escala", ["🌍 Global", "🗺️ Por país"], horizontal=True)

    series = [("Temperatura (°C)", COL_TMP, "#c0392b"),
              ("Emisiones de CO₂ (t)", COL_CO2, "#5d3a9b"),
              ("Consumo fósil (TWh)", COL_FOS, "#2a6f7f")]

    ver_svr = st.checkbox("Mostrar también ajuste SVR",
                          help="El SVR (kernel RBF) sigue la curva de forma más "
                               "flexible, pero no extrapola bien fuera del rango.")

    if modo == "Global":
        gg = df.groupby("Anio").agg(**{COL_TMP: (COL_TMP, "mean"),
                                       COL_CO2: (COL_CO2, "sum"),
                                       COL_FOS: (COL_FOS, "sum")}).reset_index()
        st.caption("Temperatura = promedio de los países · CO₂ y fósil = suma mundial.")
        base_data = gg
        titulo_extra = "(global)"
    else:
        pais = st.selectbox("País", sorted(df["Pais"].unique()))
        base_data = df[df["Pais"] == pais].sort_values("Anio")
        titulo_extra = f"({pais})"

    cols = st.columns(3)
    for cc, (nombre, col, color) in zip(cols, series):
        d = base_data.dropna(subset=[col])
        m = LinearRegression().fit(d[["Anio"]], d[col])
        tend = m.predict(d[["Anio"]])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["Anio"], y=d[col], mode="lines+markers",
                                 line=dict(color=color), name="Observado",
                                 marker=dict(size=4)))
        fig.add_trace(go.Scatter(x=d["Anio"], y=tend, mode="lines",
                                 line=dict(color="black", dash="dash"),
                                 name="Tendencia lineal"))
        if ver_svr:
            svr = make_pipeline(StandardScaler(),
                                SVR(kernel="rbf", C=10, gamma="scale")).fit(d[["Anio"]], d[col])
            fig.add_trace(go.Scatter(x=d["Anio"], y=svr.predict(d[["Anio"]]),
                                     mode="lines", line=dict(color="#2e8b57", dash="dot",
                                                             width=3), name="Ajuste SVR"))
        signo = "↑" if m.coef_[0] > 0 else "↓"
        fig.update_layout(title=f"{nombre} {signo}", height=380,
                          showlegend=ver_svr, xaxis_title="", yaxis_title="",
                          legend=dict(orientation="h", y=-0.2))
        cc.plotly_chart(fig, use_container_width=True)

    st.caption(f"Línea negra punteada = tendencia lineal {titulo_extra}. "
               "La flecha ↑/↓ indica el signo de la pendiente.")


# ======================================================================
# 6. COMPARACIÓN DE MODELOS (Lineal vs SVR)
# ======================================================================
elif seccion == "Comparación de modelos":
    st.title("Selección del mejor modelo")
    st.markdown("Comparamos **4 modelos** para predecir la temperatura por país, con "
                "validación temporal común: entrenamiento **1980–2018**, prueba "
                "**2019–2024**. Se agregan las predicciones de todos los países.")

    from sklearn.svm import SVR  # noqa: F811
    from sklearn.metrics import mean_absolute_error
    modelos = ["Regresión lineal", "SVR ajustado", "Media móvil (10 años)", "Naive"]
    preds = {m: [] for m in modelos}
    y_real = []
    for pais, d in df[df["Anio"].between(1980, 2024)].groupby("Pais"):
        d = d.sort_values("Anio")
        tr, te = d[d["Anio"] <= 2018], d[d["Anio"] > 2018]
        if len(te) < 2 or len(tr) < 10:
            continue
        y_real += list(te[COL_TMP])
        lin = LinearRegression().fit(tr[["Anio"]], tr[COL_TMP])
        preds["Regresión lineal"] += list(lin.predict(te[["Anio"]]))
        svr = make_pipeline(StandardScaler(),
                            SVR(kernel="rbf", C=10, gamma="scale")).fit(tr[["Anio"]], tr[COL_TMP])
        preds["SVR ajustado"] += list(svr.predict(te[["Anio"]]))
        preds["Media móvil (10 años)"] += [tr[COL_TMP].iloc[-10:].mean()] * len(te)
        preds["Naive"] += [tr[COL_TMP].iloc[-1]] * len(te)

    y_real = np.array(y_real)
    met = []
    for m in modelos:
        p = np.array(preds[m])
        met.append({"Modelo": m,
                    "RMSE": np.sqrt(np.mean((y_real - p) ** 2)),
                    "MAE": mean_absolute_error(y_real, p),
                    "R2": r2_score(y_real, p)})
    met = pd.DataFrame(met)
    colores = {"Regresión lineal": "#3b3b6d", "SVR ajustado": "#2a6f7f",
               "Media móvil (10 años)": "#2e8b57", "Naive": "#7cb342"}

    st.subheader("Desempeño en datos no usados para el ajuste (2019–2024)")
    cols = st.columns(3)
    for cc, metrica in zip(cols, ["RMSE", "MAE", "R2"]):
        fig = px.bar(met, x="Modelo", y=metrica, color="Modelo",
                     color_discrete_map=colores, title=f"Comparación de {metrica}")
        fig.update_layout(showlegend=False, xaxis_title="", xaxis_tickangle=-25,
                          height=380)
        cc.plotly_chart(fig, use_container_width=True)

    mejor = met.loc[met["RMSE"].idxmin(), "Modelo"]
    rmse_mejor = met["RMSE"].min()
    st.success(f"**Modelo elegido: {mejor}** — obtuvo el menor RMSE "
               f"({rmse_mejor:.4f} °C) en el período de prueba. Captura la tendencia "
               f"de calentamiento mejor que los baseline (naive y media móvil), y a "
               f"diferencia del SVR extrapola de forma confiable más allá de 2024.")
    with st.expander("Ver tabla de métricas"):
        st.dataframe(met.round(4), use_container_width=True, hide_index=True)


# ======================================================================
# 7. EXPLORACIÓN INICIAL (KAGGLE)
# ======================================================================
elif seccion == "Exploración inicial (Kaggle)":
    st.title("Exploración inicial · dataset Kaggle")
    st.info("Este era el **dataset sintético inicial** que se usó para la fase de "
            "decisión. Se descartó a favor de la base real (sus valores son aleatorios, "
            "sin tendencia). Se incluye para documentar el proceso.")

    if kaggle is None:
        st.error("No se encontró `Kaggle_Climate_Dataset.csv` en el repositorio. "
                 "Súbelo junto a `app.py` para ver esta sección.")
    else:
        k = kaggle.copy()
        graf = st.selectbox("Gráfico", [
            "Distribución de variables (boxplots)",
            "Umbrales de CO₂ según escenario 2050",
            "Cuotas de consumo fósil según escenario 2050",
            "Escenarios de riesgo por país",
            "Baseline de temperatura global",
            "Comparación Kaggle vs base real (OWID)",
        ])

        if graf == "Distribución de variables (boxplots)":
            vs = ["global_avg_temperature", "temperature_anomaly", "co2_concentration_ppm",
                  "fossil_fuel_consumption", "renewable_energy_share", "sea_level_rise_mm",
                  "sea_surface_temperature", "heatwave_days", "drought_index",
                  "climate_risk_index"]
            sel = st.multiselect("Variables", vs, default=vs[:6])
            for fila in range(0, len(sel), 3):
                cols = st.columns(3)
                for cc, var in zip(cols, sel[fila:fila + 3]):
                    f = px.box(k, y=var, points="outliers",
                               color_discrete_sequence=["#2e6b9a"], title=var)
                    f.update_layout(height=320, showlegend=False, yaxis_title="")
                    cc.plotly_chart(f, use_container_width=True)

        elif graf.startswith("Umbrales de CO₂"):
            k["temp_red"] = np.round(k["predicted_temperature_2050"] * 2) / 2
            kf = k[(k["temp_red"] >= 1.0) & (k["temp_red"] <= 4.0)]
            fig = px.box(kf, x="temp_red", y="co2_concentration_ppm",
                         facet_col="country", facet_col_wrap=5,
                         color_discrete_sequence=["#708090"],
                         labels={"temp_red": "Temp. proyectada 2050 (°C)",
                                 "co2_concentration_ppm": "CO₂ (ppm)"})
            fig.update_layout(height=650,
                              title="Umbrales de CO₂ según escenario de temperatura 2050")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)

        elif graf.startswith("Cuotas de consumo"):
            k["temp_red"] = np.round(k["predicted_temperature_2050"] * 2) / 2
            kf = k[(k["temp_red"] >= 1.0) & (k["temp_red"] <= 4.0)]
            fig = px.box(kf, x="temp_red", y="fossil_fuel_consumption",
                         facet_col="country", facet_col_wrap=5,
                         color_discrete_sequence=["#8a5a44"],
                         labels={"temp_red": "Temp. proyectada 2050 (°C)",
                                 "fossil_fuel_consumption": "Consumo fósil"})
            fig.update_layout(height=650,
                              title="Cuotas de consumo fósil según escenario 2050")
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)

        elif graf.startswith("Escenarios de riesgo"):
            umbral = k["predicted_temperature_2050"].median()
            k["Riesgo"] = np.where(k["predicted_temperature_2050"] > umbral,
                                   "Severo (> mediana)", "Moderado (< mediana)")
            conteo = (k.groupby(["country", "Riesgo"]).size()
                      .reset_index(name="n"))
            fig = px.bar(conteo, x="country", y="n", color="Riesgo", barmode="stack",
                         color_discrete_map={"Severo (> mediana)": "#b33030",
                                             "Moderado (< mediana)": "#708090"},
                         labels={"country": "País", "n": "N° de registros"},
                         title="Distribución de escenarios de riesgo térmico por país")
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Umbral = mediana de la temperatura proyectada 2050 "
                       f"({umbral:.2f} °C).")

        else:  # Baseline global
            gh = k.groupby("year")["global_avg_temperature"].mean()
            m_tr = LinearRegression().fit(gh.index.values.reshape(-1, 1), gh.values)
            fut_b = np.arange(int(gh.index.max()) + 1, 2051)
            naive, ma10 = gh.iloc[-1], gh.iloc[-10:].mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=gh.index, y=gh.values, mode="lines",
                                     line=dict(color="black"), name="Histórico"))
            fig.add_trace(go.Scatter(x=fut_b, y=[naive] * len(fut_b), mode="lines",
                                     line=dict(color="#c0392b", dash="dash"),
                                     name=f"Naive ({naive:.2f})"))
            fig.add_trace(go.Scatter(x=fut_b, y=[ma10] * len(fut_b), mode="lines",
                                     line=dict(color="#2e8b57", dash="dash"),
                                     name=f"Media móvil 10y ({ma10:.2f})"))
            fig.add_trace(go.Scatter(x=fut_b, y=m_tr.predict(fut_b.reshape(-1, 1)),
                                     mode="lines", line=dict(color="#2471a3", width=3),
                                     name="Tendencia lineal"))
            fig.update_layout(xaxis_title="Año", yaxis_title="Temperatura global (°C)",
                              title="Baseline y tendencia (dataset Kaggle)",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Al ser datos sintéticos, la tendencia es casi plana — otra "
                       "razón por la que se optó por la base real.")

        # ------- Comparación Kaggle vs base real (OWID) -------
        if graf.startswith("Comparación"):
            st.markdown("Compara la **forma** de cada variable entre el dataset Kaggle "
                        "(sintético) y la base real (OWID), por país. Como las unidades "
                        "difieren, ambas series se **normalizan a 0–1**.")
            k2 = kaggle.copy()
            k2["country"] = k2["country"].replace({"UK": "United Kingdom",
                                                   "USA": "United States"})
            c1, c2 = st.columns(2)
            pais = c1.selectbox("País", sorted(df["Pais"].unique()), key="cmp_pais")
            var_map = {
                "Temperatura": (COL_TMP, "global_avg_temperature"),
                "CO₂": (COL_CO2, "co2_concentration_ppm"),
                "Consumo fósil": (COL_FOS, "fossil_fuel_consumption"),
            }
            var = c2.selectbox("Variable", list(var_map.keys()), key="cmp_var")
            col_real, col_kag = var_map[var]

            def norm(s):
                s = s.astype(float)
                return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else s * 0

            dr = df[df["Pais"] == pais].sort_values("Anio")
            dk = (k2[k2["country"] == pais].groupby("year")[col_kag]
                  .mean().reset_index())

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dr["Anio"], y=norm(dr[col_real]),
                                     mode="lines+markers", name="Base real (OWID)",
                                     line=dict(color="#2471a3", width=3),
                                     marker=dict(size=4)))
            fig.add_trace(go.Scatter(x=dk["year"], y=norm(dk[col_kag]),
                                     mode="lines+markers", name="Kaggle (sintético)",
                                     line=dict(color="#c0392b", dash="dot"),
                                     marker=dict(size=4)))
            fig.update_layout(title=f"{var} en {pais} · normalizado (0–1)",
                              xaxis_title="Año", yaxis_title="Valor normalizado",
                              legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            st.info("La serie azul (real/OWID) muestra una tendencia clara; la roja "
                    "(Kaggle) es plana y ruidosa. Esta comparación es la que justificó "
                    "usar la base real para el análisis.")


# ======================================================================
# 8. CONCLUSIONES
# ======================================================================
elif seccion == "Conclusiones":
    st.title("Conclusiones")
    st.markdown(
        """
- **Existe una relación fuerte entre emisiones y temperatura a escala global**
  (correlación ≈ 0.89, R² ≈ 0.80). Por cada gigatonelada extra de CO₂, la anomalía
  sube alrededor de 0.09 °C.

- **CO₂ y consumo de combustibles fósiles son intercambiables** (correlación 0.99):
  quemar fósiles genera el CO₂. Por eso ambos modelos dan resultados casi idénticos,
  y no deben combinarse en una misma regresión.

- **A nivel país la relación se rompe**: Francia, Alemania y Reino Unido muestran
  correlación *negativa* porque bajaron sus emisiones y aun así se calentaron. La
  temperatura de un país responde al CO₂ **global**, no al propio.

- **Proyección a 2050**: bajo un escenario de tendencia lineal, la anomalía llegaría
  a ≈ +1.7 °C sobre la base 1980–2000. Es un escenario, no una predicción dura: el
  backtest muestra que la trayectoria de emisiones es la mayor fuente de incertidumbre.
        """
    )
    st.caption("Proyecto Integrador — Ciencia de Datos Ambientales.")
