import datetime
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st


MODULO1 = "Módulo 1 - Home"
MODULO2 = "Módulo 2 - Análisis Exploratorio de Datos"
MODULO3 = "Módulo 3 - Conclusiones finales"


class AnalizadorDatos:
    def __init__(self, datos):
        self.datos = datos

    def clasificar_variables(self):
        columnas_numericas = self.datos.select_dtypes(include="number").columns.tolist()
        columnas_categoricas = self.datos.select_dtypes(include=["object", "category"]).columns.tolist()
        return columnas_numericas, columnas_categoricas

    def describir_numericas(self):
        columnas_numericas, _ = self.clasificar_variables()
        return self.datos[columnas_numericas].describe() if columnas_numericas else pd.DataFrame()

    def describir_categoricas(self):
        _, columnas_categoricas = self.clasificar_variables()
        return self.datos[columnas_categoricas].describe() if columnas_categoricas else pd.DataFrame()

    def valores_faltantes(self):
        return self.datos.isna().sum().sort_values(ascending=False)

    def numerica_por_categoria(self, columna_numerica, columna_categorica):
        datos_agrupados = self.datos.groupby(columna_categorica, dropna=False)[columna_numerica]
        resumenes = []
        for categoria, valores in datos_agrupados:
            observaciones = valores.dropna().to_numpy(dtype=float)
            resumenes.append(
                {
                    "Grupo": categoria,
                    "conteo": observaciones.size,
                    "media": np.mean(observaciones) if observaciones.size else np.nan,
                    "mediana": np.median(observaciones) if observaciones.size else np.nan,
                    "desviacion": np.std(observaciones, ddof=1) if observaciones.size > 1 else np.nan,
                }
            )
        return pd.DataFrame(resumenes).set_index("Grupo").round(2)

    def distribucion_categorica(self, columna):
        conteos = self.datos[columna].value_counts(dropna=False)
        proporciones = (conteos / len(self.datos) * 100).round(2)
        return pd.DataFrame({"Conteo": conteos, "Proporción (%)": proporciones})

    def graficar_histograma(self, columna, intervalos, mostrar_cuadricula):
        figura, eje = plt.subplots(figsize=(9, 4.5))
        sns.histplot(
            data=self.datos,
            x=columna,
            bins=intervalos,
            color="#176B87",
            edgecolor="white",
            ax=eje,
        )
        eje.set_title(f"Distribución de {columna}")
        eje.set_xlabel(columna)
        eje.set_ylabel("Frecuencia")
        eje.grid(mostrar_cuadricula, axis="y", alpha=0.25)
        figura.tight_layout()
        return figura

    def graficar_distribucion_categorica(self, columna, cantidad):
        conteos = self.datos[columna].value_counts(dropna=False).head(cantidad).sort_values()
        datos_grafico = pd.DataFrame(
            {"Categoría": conteos.index.astype(str), "Conteo": conteos.to_numpy()}
        )
        figura, eje = plt.subplots(figsize=(9, 4.5))
        sns.barplot(
            data=datos_grafico,
            x="Conteo",
            y="Categoría",
            color="#D97745",
            errorbar=None,
            ax=eje,
        )
        eje.set_title(f"Frecuencia de {columna}")
        eje.set_xlabel("Conteo")
        eje.set_ylabel(columna)
        figura.tight_layout()
        return figura

    def graficar_numerica_por_categoria(self, columna_numerica, columna_categorica):
        figura, eje = plt.subplots(figsize=(9, 4.5))
        sns.boxplot(
            data=self.datos,
            x=columna_categorica,
            y=columna_numerica,
            showfliers=False,
            ax=eje,
        )
        eje.set_title(f"{columna_numerica} según {columna_categorica}")
        eje.set_xlabel(columna_categorica)
        eje.set_ylabel(columna_numerica)
        eje.tick_params(axis="x", rotation=30)
        figura.tight_layout()
        return figura


def mostrar_grafico(figura):
    st.pyplot(figura, width="stretch")
    plt.close(figura)


def mostrar_hallazgos(analizador):
    datos = analizador.datos
    hallazgos = []
    total_faltantes = int(datos.isna().sum().sum())
    hallazgos.append(
        f"El dataset contiene {len(datos):,} registros y {datos.shape[1]} variables; "
        f"se encontraron {total_faltantes:,} valores nulos."
    )

    _, columnas_categoricas = analizador.clasificar_variables()

    if "y" in datos.columns:
        objetivo = datos["y"].astype("string").str.lower()
        objetivo_valido = objetivo.dropna()
        if not objetivo_valido.empty:
            aceptaciones = objetivo_valido.eq("yes")
            hallazgos.append(
                f"La campaña fue aceptada en {aceptaciones.mean() * 100:.2f}% de los registros "
                f"con resultado informado ({int(aceptaciones.sum()):,} de {len(objetivo_valido):,})."
            )

            if "job" in datos.columns:
                resultados_ocupacion = datos.assign(
                    _aceptado=objetivo.eq("yes").where(objetivo.notna())
                ).groupby(
                    "job", dropna=False
                )["_aceptado"]
                aceptacion_por_ocupacion = resultados_ocupacion.mean().dropna()
                if not aceptacion_por_ocupacion.empty:
                    ocupacion_mayor = aceptacion_por_ocupacion.idxmax()
                    ocupacion_menor = aceptacion_por_ocupacion.idxmin()
                    hallazgos.append(
                        f"Por ocupación, la mayor tasa observada fue {aceptacion_por_ocupacion[ocupacion_mayor] * 100:.2f}% "
                        f"en {ocupacion_mayor} (n={int(resultados_ocupacion.count().loc[ocupacion_mayor])}); "
                        f"la menor fue {aceptacion_por_ocupacion[ocupacion_menor] * 100:.2f}% en {ocupacion_menor} "
                        f"(n={int(resultados_ocupacion.count().loc[ocupacion_menor])})."
                    )

            if "duration" in datos.columns:
                duracion_por_resultado = datos.assign(_resultado=objetivo).groupby("_resultado")["duration"].mean()
                if "yes" in duracion_por_resultado.index and "no" in duracion_por_resultado.index:
                    diferencia = duracion_por_resultado["yes"] - duracion_por_resultado["no"]
                    direccion = "mayor" if diferencia > 0 else "menor" if diferencia < 0 else "igual"
                    hallazgos.append(
                        f"La duración media de los contactos con aceptación fue {abs(diferencia):,.2f} segundos "
                        f"{direccion} que la de los contactos sin aceptación."
                    )

    if columnas_categoricas:
        columna = "job" if "job" in columnas_categoricas else columnas_categoricas[0]
        conteos = datos[columna].value_counts(dropna=False)
        if not conteos.empty:
            categoria_frecuente = conteos.index[0]
            hallazgos.append(
                f"La categoría más frecuente de {columna} es {categoria_frecuente}, con {int(conteos.iloc[0]):,} registros."
            )

    for hallazgo in hallazgos[:5]:
        st.markdown(f"- {hallazgo}")


st.set_page_config(
    page_title="Análisis de campañas bancarias",
    layout="wide",
    page_icon="📊",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #f4f7f8; }
    [data-testid="stHeader"] { background-color: rgba(244, 247, 248, 0.92); }
    h1, h2, h3 { color: #123047; }
    [data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d8e2e7;
        border-left: 4px solid #176B87;
        padding: 0.9rem 1rem;
        border-radius: 4px;
    }
    [data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("img/python-logo.png", width=300)
st.title("Análisis de campañas bancarias")
st.caption("Especialización en Python for Analytics | Análisis exploratorio de datos")

st.sidebar.image("img/python-for-analytics.png", use_container_width=True)
opcion = st.sidebar.selectbox("Seleccione una sección", (MODULO1, MODULO2, MODULO3), index=0)

if opcion == MODULO1:
    st.header("🎯 Presentación del proyecto")
    
    st.subheader("🔥 Análisis exploratorio de BankMarketing")
    st.write(
        "Aplicación interactiva para explorar los datos de una campaña bancaria y "
        "examinar relaciones y comportamientos entre sus variables."
    )
    st.write("El objetivo es aplicar análisis descriptivo para apoyar la toma de decisiones, no construir modelos predictivos.")  

    st.divider()
    #columna_autor, columna_curso, columna_anio = st.columns(3)
    #columna_autor.metric("Autor", "Anibal Abraham Minaya Cubillas")
    #columna_curso.metric("Curso", "Módulo 1: Python Fundamentals")
    #columna_anio.metric("Año", datetime.datetime.now().year)
    st.metric("Autor", "Anibal Abraham Minaya Cubillas")
    st.metric("Curso", "Módulo 2: Python for Analytics")
    st.metric("Año", datetime.datetime.now().year)
    st.subheader("Tecnologías utilizadas")
    columnas_tecnologias = st.columns(4)
    columnas_tecnologias[0].image("img/python-logo.png", width=120)
    columnas_tecnologias[1].image("img/streamlit.png", width=150)
    columnas_tecnologias[2].image("img/github.png", width=120)
    columnas_tecnologias[3].image("img/seaborn.jfif", width=120)
    columnas_tecnologias = st.columns(3)
    columnas_tecnologias[0].image("img/pandas.png", width=120)
    columnas_tecnologias[1].image("img/numpy.png", width=150)
    columnas_tecnologias[2].image("img/matplotlib.jfif", width=120)

elif opcion == MODULO2:
    st.header("Análisis Exploratorio de Datos")
    st.write("Carga el archivo CSV para consultar las estadísticas y visualizaciones del dataset.")
    archivo_cargado = st.file_uploader("Selecciona BankMarketing.csv", type=["csv"])

    if archivo_cargado is not None or "datos_cargados" in st.session_state:
        try:
            if archivo_cargado is not None:
                datos = pd.read_csv(archivo_cargado, sep=";")
                nombre_archivo = archivo_cargado.name
            else:
                datos = st.session_state["datos_cargados"]
                nombre_archivo = st.session_state.get("nombre_archivo", "CSV cargado previamente")
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as excepcion:
            st.error(f"No se pudo leer el archivo CSV: {excepcion}")
        else:
            if datos.empty or len(datos.columns) == 0:
                st.error("El archivo no contiene registros ni columnas para analizar.")
            else:
                st.session_state["datos_cargados"] = datos
                st.session_state["nombre_archivo"] = nombre_archivo
                st.success(f"Archivo cargado: {nombre_archivo}")
                columna_vista_previa, columna_dimensiones = st.columns([2, 1])
                with columna_vista_previa:
                    st.subheader("Vista previa")
                    st.dataframe(datos.head(), width="stretch", hide_index=True)
                with columna_dimensiones:
                    st.subheader("Dimensiones")
                    columnas_metricas = st.columns(2)
                    columnas_metricas[0].metric("Filas", f"{datos.shape[0]:,}")
                    columnas_metricas[1].metric("Columnas", datos.shape[1])

                analizador = AnalizadorDatos(datos)
                columnas_numericas, columnas_categoricas = analizador.clasificar_variables()
                
                st.subheader("Tabs con análisis detallado")
                # Mostrar los tabs de Items
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
                    [
                        "1. Información",
                        "2. Variables",
                        "3. Estadísticas",
                        "4. Faltantes",
                        "5. Numéricas",
                        "6. Categóricas",
                        "7. Numérica vs categórica",
                        "8. Categórica vs categórica",
                        "9. Análisis dinámico",
                        "10. Resumen",
                    ]
                )

                with tab1:
                    st.header("Información general del dataset")
                    memoria_texto = io.StringIO()
                    datos.info(buf=memoria_texto)
                    st.subheader(".info()")
                    st.code(memoria_texto.getvalue())
                    st.subheader("Tipos de datos")
                    st.dataframe(datos.dtypes.astype(str).rename("Tipo"), width="content", height="content")
                    st.subheader("Conteo de valores nulos")
                    st.dataframe(analizador.valores_faltantes().rename("Valores nulos"), width="content", height="content")

                with tab2:
                    st.header("Clasificación de variables")
                    st.write("La clasificación se obtiene con una función personalizada según el tipo de dato.")
                    columna_numericas, columna_categoricas = st.columns(2)
                    with columna_numericas:
                        st.subheader("Variables numéricas")
                        st.dataframe(pd.DataFrame({"Variable": columnas_numericas}), width="stretch", hide_index=True)
                    with columna_categoricas:
                        st.subheader("Variables categóricas")
                        st.dataframe(pd.DataFrame({"Variable": columnas_categoricas}), width="stretch", hide_index=True)
                    columnas_conteo = st.columns(2)
                    columnas_conteo[0].metric("Numéricas", len(columnas_numericas))
                    columnas_conteo[1].metric("Categóricas", len(columnas_categoricas))

                with tab3:
                    st.header("Estadísticas descriptivas")
                    st.subheader("Variables numéricas")
                    descripcion_numericas = analizador.describir_numericas()
                    if not descripcion_numericas.empty:
                        st.dataframe(descripcion_numericas, width="stretch")
                        st.caption("La media resume el promedio; la mediana corresponde al percentil 50 y la desviación estándar describe la dispersión.")
                    else:
                        st.info("No hay variables numéricas en el archivo.")
                    st.subheader("Variables categóricas")
                    descripcion_categoricas = analizador.describir_categoricas()
                    if not descripcion_categoricas.empty:
                        st.dataframe(descripcion_categoricas, width="stretch")
                        st.caption("El resumen muestra el conteo, los valores distintos, la categoría más frecuente y su frecuencia.")
                    else:
                        st.info("No hay variables categóricas en el archivo.")

                with tab4:
                    st.header("Análisis de valores faltantes")
                    faltantes = analizador.valores_faltantes()
                    faltantes = faltantes[faltantes > 0]
                    if faltantes.empty:
                        st.success("No se encontraron valores faltantes en el dataset.")
                    else:
                        st.dataframe(faltantes.rename("Valores nulos"), width="stretch")
                        st.bar_chart(faltantes.rename("Valores nulos"), horizontal=True)
                        st.write(
                            f"Los valores faltantes se concentran en {len(faltantes)} variable(s), "
                            f"con {int(faltantes.sum()):,} celdas vacías en total."
                        )

                with tab5:
                    st.header("Distribución de variables numéricas")
                    if columnas_numericas:
                        columna_seleccionada = st.selectbox("Variable numérica", columnas_numericas, key="hist_column")
                        intervalos = st.slider("Número de intervalos", min_value=5, max_value=50, value=20)
                        mostrar_cuadricula = st.checkbox("Mostrar cuadrícula", value=True)
                        mostrar_grafico(analizador.graficar_histograma(columna_seleccionada, intervalos, mostrar_cuadricula))
                        st.write(
                            f"El histograma muestra la distribución observada de {columna_seleccionada}; "
                            "la forma y concentración se aprecian directamente en el gráfico."
                        )
                    else:
                        st.info("No hay variables numéricas para graficar.")

                with tab6:
                    st.header("Análisis de variables categóricas")
                    if columnas_categoricas:
                        categoria_seleccionada = st.selectbox("Variable categórica", columnas_categoricas, key="category_column")
                        cantidad_categorias = max(1, int(datos[categoria_seleccionada].nunique(dropna=False)))
                        cantidad_mostrar = st.slider("Categorías a mostrar", 1, min(20, cantidad_categorias), min(10, cantidad_categorias))
                        col1,col2 = st.columns(2)
                        col1.dataframe(analizador.distribucion_categorica(categoria_seleccionada), width="content", height="content")
                        mostrar_grafico(analizador.graficar_distribucion_categorica(categoria_seleccionada, cantidad_mostrar))
                    else:
                        st.info("No hay variables categóricas para analizar.")

                with tab7:
                    st.header("Análisis bivariado: numérico vs categórico")
                    if columnas_numericas and columnas_categoricas:
                        categoria_preferida = "y" if "y" in columnas_categoricas else columnas_categoricas[0]
                        columna_numerica_seleccionada = st.selectbox("Variable numérica", columnas_numericas, key="bivar_numeric")
                        columna_categorica_seleccionada = st.selectbox(
                            "Variable categórica", columnas_categoricas,
                            index=columnas_categoricas.index(categoria_preferida), key="bivar_category"
                        )
                        comparacion = analizador.numerica_por_categoria(columna_numerica_seleccionada, columna_categorica_seleccionada)
                        st.dataframe(comparacion, width="stretch")
                        mostrar_grafico(analizador.graficar_numerica_por_categoria(columna_numerica_seleccionada, columna_categorica_seleccionada))
                    else:
                        st.info("Se requieren variables numéricas y categóricas para esta comparación.")

                with tab8:
                    st.header("Análisis bivariado: categórico vs categórico")
                    if len(columnas_categoricas) >= 2:
                        primera_categoria = "education" if "education" in columnas_categoricas else columnas_categoricas[0]
                        segunda_categoria = "y" if "y" in columnas_categoricas else next(
                            columna for columna in columnas_categoricas if columna != primera_categoria
                        )
                        indice_primera = columnas_categoricas.index(primera_categoria)
                        categoria_filas = st.selectbox("Variable de filas", columnas_categoricas, index=indice_primera, key="cross_row")
                        columnas_disponibles = [columna for columna in columnas_categoricas if columna != categoria_filas]
                        categoria_columnas = st.selectbox(
                            "Variable de columnas", columnas_disponibles,
                            index=columnas_disponibles.index(segunda_categoria) if segunda_categoria in columnas_disponibles else 0,
                            key="cross_column"
                        )
                        conteos_cruzados = pd.crosstab(datos[categoria_filas], datos[categoria_columnas], dropna=False)
                        proporciones = pd.crosstab(
                            datos[categoria_filas], datos[categoria_columnas], normalize="index", dropna=False
                        ).mul(100).round(2)
                        st.subheader("Conteos")
                        st.dataframe(conteos_cruzados, width="stretch")
                        st.subheader("Proporciones por fila (%)")
                        st.dataframe(proporciones, width="stretch")
                        st.bar_chart(proporciones, stack=False)
                    else:
                        st.info("Se requieren al menos dos variables categóricas.")

                with tab9:
                    st.header("Análisis basado en parámetros seleccionados")
                    columnas_seleccionadas = st.multiselect(
                        "Selecciona las variables que deseas analizar",
                        datos.columns.tolist(),
                        default=datos.columns[:min(3, len(datos.columns))].tolist(),
                    )
                    if columnas_seleccionadas:
                        datos_seleccionados = datos[columnas_seleccionadas]
                        datos_numericos = datos_seleccionados.select_dtypes(include="number")
                        datos_categoricos = datos_seleccionados.select_dtypes(include=["object", "category"])
                        if not datos_numericos.empty:
                            st.subheader("Resumen numérico")
                            st.dataframe(datos_numericos.describe(), width="content", height="content")
                        for columna in datos_categoricos.columns:
                            st.subheader(f"Conteo de {columna}")
                            st.dataframe(
                                datos_categoricos[columna].value_counts(dropna=False).rename("Conteo"),
                                width="content", height="content",
                            )
                    else:
                        st.info("Selecciona al menos una variable para mostrar su análisis.")

                with tab10:
                    st.header("Hallazgos clave")
                    st.subheader("Resumen visual")
                    if "y" in datos.columns:
                        conteo_objetivo = datos["y"].value_counts(dropna=False)
                        st.bar_chart(conteo_objetivo.rename("Registros"))
                    if "job" in datos.columns and "y" in datos.columns:
                        aceptacion = datos.assign(
                            _aceptado=datos["y"].astype("string").str.lower().eq("yes").where(
                                datos["y"].notna()
                            )
                        ).groupby("job", dropna=False)["_aceptado"].mean().mul(100).sort_values(ascending=False)
                        st.bar_chart(aceptacion.rename("Aceptación (%)"))
else:
    st.header("💡 Conclusiones finales")
    st.write(
        "Conclusiones descriptivas derivadas de los análisis del dataset, "
        "como insumo para la toma de decisiones y no como predicciones."
    )
    if "datos_cargados" in st.session_state:
        mostrar_hallazgos(AnalizadorDatos(st.session_state["datos_cargados"]))
    else:
        st.info("Carga primero el archivo CSV en Módulo 2 para generar las conclusiones.")