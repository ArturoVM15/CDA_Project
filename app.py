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
                   page_icon="🌍", layout="wide")

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


df = cargar_datos()
g = serie_global(df)

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
st.sidebar.title("🌍 Navegación")
seccion = st.sidebar.radio(
    "Ir a:",
    ["🏠 Inicio", "🔍 Exploración (EDA)", "🌡️ Relación global",
     "🗺️ Relación por país", "📊 Matriz predictiva",
     "📈 Proyección a 2050", "🏆 Comparación de modelos",
     "📝 Conclusiones"]
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Datos: {len(df)} registros · "
                   f"{df['Pais'].nunique()} países · "
                   f"{int(df['Anio'].min())}–{int(df['Anio'].max())}")


# ======================================================================
# 1. INICIO
# ======================================================================
if seccion == "🏠 Inicio":
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
            "relación (global y por país), proyección a 2050 y clasificación KNN.")


# ======================================================================
# 2. EXPLORACIÓN (EDA)
# ======================================================================
elif seccion == "🔍 Exploración (EDA)":
    st.title("🔍 Análisis exploratorio")

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
elif seccion == "🌡️ Relación global":
    st.title("🌡️ Relación a nivel global")
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
elif seccion == "🗺️ Relación por país":
    st.title("🗺️ Relación por país")
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
elif seccion == "📈 Proyección a 2050":
    st.title("📈 Proyección de calentamiento a 2050")

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

        st.metric(f"{etiqueta} proyectada a 2050",
                  f"{modelo.predict(pd.DataFrame({'Anio':[2050]}))[0]:,.2f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d["Anio"], y=d[col], mode="lines+markers",
                                 line=dict(color="#2471a3"), name="Histórico"))
        fig.add_trace(go.Scatter(x=fut["Anio"], y=pred, mode="lines",
                                 line=dict(color="#c0392b", dash="dash"),
                                 name="Proyección"))
        fig.add_vline(x=int(d["Anio"].max()), line_dash="dot", line_color="gray")
        fig.update_layout(xaxis_title="Año", yaxis_title=etiqueta,
                          title=f"{pais} · proyección lineal a 2050",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# 5b. MATRIZ PREDICTIVA 4x4
# ======================================================================
elif seccion == "📊 Matriz predictiva":
    st.title("📊 Matriz predictiva de temperatura a 2050")
    st.markdown("Un modelo de relación predice la anomalía de temperatura para cada "
                "combinación de **nivel de CO₂ × nivel de consumo fósil**, proyectada a 2050.")

    # Modelo de relación: anomalía ~ CO2 + fósil + año
    Xg = df[[COL_CO2, COL_FOS, "Anio"]]
    modelo = LinearRegression().fit(Xg, df["anom"])

    niveles_emi = ["Muy Bajo", "Bajo", "Alto", "Muy Alta"]
    niveles_con = ["Mínimo", "Moderado", "Alto", "Intenso"]
    qs = [0.125, 0.375, 0.625, 0.875]
    emi_c = df[COL_CO2].quantile(qs).values
    con_c = df[COL_FOS].quantile(qs).values

    M = np.zeros((4, 4))
    for i, con in enumerate(con_c):
        for j, emi in enumerate(emi_c):
            M[i, j] = modelo.predict(pd.DataFrame(
                {COL_CO2: [emi], COL_FOS: [con], "Anio": [2050]}))[0]

    fig = px.imshow(M, x=niveles_emi, y=niveles_con, text_auto=".2f",
                    color_continuous_scale="YlOrRd", aspect="auto",
                    labels=dict(x="Nivel de Emisión de CO₂",
                                y="Intensidad de Consumo Fósil",
                                color="Anomalía 2050 (°C)"))
    fig.update_layout(title="Temperatura proyectada a 2050 según escenarios")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("Nota: como CO₂ y fósil están correlacionados al 0.99, algunas "
               "combinaciones (ej. CO₂ muy alto + consumo mínimo) son escenarios "
               "extrapolados que casi no ocurrieron en los datos.")


# ======================================================================
# 6. COMPARACIÓN DE MODELOS (Lineal vs SVR)
# ======================================================================
elif seccion == "🏆 Comparación de modelos":
    st.title("🏆 Selección del modelo: Lineal vs SVR")
    st.markdown("Comparamos **Regresión Lineal** y **SVR** para predecir la temperatura "
                "por país, con validación temporal: entrenamiento 1980–2018, "
                "prueba 2019–2024. Menor RMSE = mejor.")

    from sklearn.svm import SVR  # noqa: F811 (ya importado arriba)
    filas = []
    for pais, d in df[df["Anio"].between(1980, 2024)].groupby("Pais"):
        d = d.sort_values("Anio")
        tr, te = d[d["Anio"] <= 2018], d[d["Anio"] > 2018]
        if len(te) < 2:
            continue
        lin = LinearRegression().fit(tr[["Anio"]], tr[COL_TMP])
        svr = make_pipeline(StandardScaler(),
                            SVR(kernel="rbf", C=10, gamma="scale")).fit(tr[["Anio"]], tr[COL_TMP])
        rmse = lambda m: np.sqrt(np.mean((te[COL_TMP] - m.predict(te[["Anio"]]))**2))
        filas.append({"Pais": pais, "RMSE Lineal": rmse(lin), "RMSE SVR": rmse(svr)})
    comp = pd.DataFrame(filas)

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE promedio · Lineal", f"{comp['RMSE Lineal'].mean():.3f} °C")
    c2.metric("RMSE promedio · SVR", f"{comp['RMSE SVR'].mean():.3f} °C")
    gana = (comp["RMSE Lineal"] < comp["RMSE SVR"]).sum()
    c3.metric("Lineal gana en", f"{gana}/{len(comp)} países")

    comp_long = comp.melt(id_vars="Pais", var_name="Modelo", value_name="RMSE")
    fig = px.bar(comp_long, x="Pais", y="RMSE", color="Modelo", barmode="group",
                 color_discrete_map={"RMSE Lineal": "#2471a3", "RMSE SVR": "#c0392b"},
                 labels={"RMSE": "RMSE en prueba (°C)"})
    fig.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    st.success("La **Regresión Lineal** gana en la mayoría de los países y tiene menor "
               "RMSE promedio. Además, el SVR con kernel RBF no extrapola bien fuera del "
               "rango de años entrenado. Por eso el proyecto se queda con el modelo lineal.")
    with st.expander("Ver tabla de RMSE por país"):
        st.dataframe(comp.round(3), use_container_width=True, hide_index=True)


# ======================================================================
# 7. CONCLUSIONES
# ======================================================================
elif seccion == "📝 Conclusiones":
    st.title("📝 Conclusiones")
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

- **La clasificación KNN** alcanza ≈ 59 % de acierto (vs 33 % del azar), distinguiendo
  bien los niveles Bajo y Alto de calentamiento.
        """
    )
    st.caption("Proyecto Integrador — Ciencia de Datos Ambientales.")
