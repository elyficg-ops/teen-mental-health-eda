"""
Teen Mental Health - Análisis Exploratorio de Datos (EDA)
Autora: Elizabeth Fiorella Cotrina Gallardo
Especialización en Python for Analytics - DMC Institute - 2026
"""

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Configuración general de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Teen Mental Health EDA",
    page_icon="📊",
    layout="wide",
)

AUTOR = "Elizabeth Fiorella Cotrina Gallardo"
CURSO = "Especialización en Python for Analytics - DMC Institute"
ANIO = 2026

COLUMNAS_ESPERADAS = [
    "age", "gender", "daily_social_media_hours", "platform_usage",
    "sleep_hours", "screen_time_before_sleep", "academic_performance",
    "physical_activity", "social_interaction_level", "stress_level",
    "anxiety_level", "addiction_level", "depression_label",
]


def mostrar_fig(fig):
    """Muestra una figura de Matplotlib en Streamlit y la cierra para liberar memoria."""
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Clase principal (POO): encapsula carga, validación, estadística,
# visualización, filtros y comparaciones entre grupos.
# ---------------------------------------------------------------------------
class DataAnalyzer:
    """Clase que encapsula todo el análisis exploratorio del dataset."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # ----------------------------- Carga y validación ----------------------
    @staticmethod
    def validar_archivo(archivo) -> tuple[bool, str, pd.DataFrame | None]:
        """Lee el CSV y valida que sea utilizable.
        Devuelve (es_valido, mensaje, dataframe_o_None)."""
        try:
            df = pd.read_csv(archivo)
        except Exception as e:
            return False, f"No se pudo leer el archivo: {e}", None

        if df.empty:
            return False, "El archivo está vacío.", None

        faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
        if faltantes:
            return False, f"Faltan columnas esperadas: {', '.join(faltantes)}", None

        return True, "Archivo cargado y validado correctamente.", df

    @property
    def filas(self) -> int:
        return self.df.shape[0]

    @property
    def columnas(self) -> int:
        return self.df.shape[1]

    # ----------------------------- Ítem 1: información general -------------
    def info_texto(self) -> str:
        """Captura la salida de df.info() como texto."""
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        return buffer.getvalue()

    def resumen_tipos(self) -> pd.DataFrame:
        """Tipo de dato, no nulos y valores únicos por columna."""
        return pd.DataFrame({
            "tipo_de_dato": self.df.dtypes.astype(str),
            "no_nulos": self.df.notnull().sum(),
            "valores_unicos": self.df.nunique(),
        })

    def valores_nulos(self) -> pd.DataFrame:
        """Conteo y porcentaje de valores nulos por variable (Ítem 4)."""
        conteo = self.df.isnull().sum()
        return pd.DataFrame({
            "nulos": conteo,
            "porcentaje_%": (conteo / len(self.df) * 100).round(2),
        })

    def num_duplicados(self) -> int:
        return int(self.df.duplicated().sum())

    # ----------------------------- Ítem 2: clasificación -------------------
    def clasificar_variables(self) -> dict:
        """Función personalizada: separa variables numéricas y categóricas."""
        numericas = self.df.select_dtypes(include="number").columns.tolist()
        categoricas = [c for c in self.df.columns if c not in numericas]
        binarias = [c for c in numericas if self.df[c].nunique() == 2]
        return {"numericas": numericas, "categoricas": categoricas, "binarias": binarias}

    def columnas_numericas(self, incluir_etiqueta: bool = False) -> list:
        """Numéricas; por defecto excluye depression_label (variable binaria)."""
        cols = self.clasificar_variables()["numericas"]
        return cols if incluir_etiqueta else [c for c in cols if c != "depression_label"]

    def columnas_categoricas(self) -> list:
        return self.clasificar_variables()["categoricas"]

    # ----------------------------- Ítem 3: estadística descriptiva ---------
    def estadisticas_descriptivas(self) -> pd.DataFrame:
        """describe() ampliado con mediana, moda, rango intercuartílico y CV."""
        num = self.df[self.columnas_numericas(incluir_etiqueta=True)]
        desc = num.describe().T
        desc["mediana"] = num.median()
        desc["moda"] = num.mode().iloc[0]
        desc["IQR"] = desc["75%"] - desc["25%"]
        desc["CV_%"] = desc["std"] / desc["mean"] * 100
        return desc.round(2)

    def outliers_iqr(self) -> pd.DataFrame:
        """Detección preliminar de valores extremos con la regla 1.5 * IQR."""
        filas = []
        for col in self.columnas_numericas():
            q1, q3 = self.df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            inferior, superior = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n = int(((self.df[col] < inferior) | (self.df[col] > superior)).sum())
            filas.append({
                "variable": col,
                "limite_inferior": round(inferior, 2),
                "limite_superior": round(superior, 2),
                "n_extremos": n,
                "porcentaje_%": round(n / len(self.df) * 100, 2),
            })
        return pd.DataFrame(filas).set_index("variable")

    @staticmethod
    def _etiquetar(df: pd.DataFrame) -> pd.DataFrame:
        """Copia del DataFrame con depression_label en texto (0 - Ausencia / 1 - Presencia)."""
        df = df.copy()
        if "depression_label" in df.columns:
            df["depression_label"] = (
                df["depression_label"]
                .map({0: "0 - Ausencia", 1: "1 - Presencia"})
                .fillna(df["depression_label"].astype(str))
            )
        return df

    # ----------------------------- Frecuencias y comparaciones -------------
    def frecuencias(self, col: str) -> pd.DataFrame:
        """Conteo y proporción de cada categoría."""
        conteo = self._etiquetar(self.df)[col].value_counts()
        return pd.DataFrame({
            "conteo": conteo,
            "proporcion_%": (conteo / conteo.sum() * 100).round(2),
        })

    def comparar_grupos(self, col_num: str, col_grupo: str) -> pd.DataFrame:
        """Estadísticos de una variable numérica según los grupos de otra."""
        return (self._etiquetar(self.df).groupby(col_grupo)[col_num]
                .agg(["count", "mean", "median", "std", "min", "max"])
                .round(2))

    def tabla_cruzada(self, col_a: str, col_b: str, porcentaje: bool = True) -> pd.DataFrame:
        """Tabla cruzada entre dos categóricas (% por fila o conteos)."""
        datos = self._etiquetar(self.df)
        if porcentaje:
            return (pd.crosstab(datos[col_a], datos[col_b], normalize="index") * 100).round(2)
        return pd.crosstab(datos[col_a], datos[col_b])

    # ----------------------------- Filtros (Ítem 9) ------------------------
    def filtrar(self, edad=None, generos=None, plataformas=None, interacciones=None) -> pd.DataFrame:
        """Devuelve el DataFrame filtrado según los parámetros recibidos."""
        mascara = pd.Series(True, index=self.df.index)
        if edad is not None:
            mascara &= self.df["age"].between(edad[0], edad[1])
        if generos:
            mascara &= self.df["gender"].isin(generos)
        if plataformas:
            mascara &= self.df["platform_usage"].isin(plataformas)
        if interacciones:
            mascara &= self.df["social_interaction_level"].isin(interacciones)
        return self.df[mascara]

    # ----------------------------- Visualizaciones -------------------------
    @staticmethod
    def _figura(ancho=6, alto=4):
        fig, ax = plt.subplots(figsize=(ancho, alto))
        return fig, ax

    def histograma(self, col: str, df=None, bins: int = 20, kde: bool = True):
        datos = self.df if df is None else df
        fig, ax = self._figura()
        sns.histplot(datos[col], bins=bins, kde=kde, ax=ax, color="#3366FF")
        ax.axvline(datos[col].mean(), color="red", linestyle="--", label="Media")
        ax.axvline(datos[col].median(), color="green", linestyle="-", label="Mediana")
        ax.set_title(f"Distribución de {col}")
        ax.legend()
        fig.tight_layout()
        return fig

    def grafico_barras(self, col: str, df=None):
        datos = self._etiquetar(self.df if df is None else df)
        fig, ax = self._figura()
        orden = datos[col].value_counts().index
        sns.countplot(data=datos, x=col, order=orden, ax=ax, palette="Blues_r", hue=col, legend=False)
        for contenedor in ax.containers:
            ax.bar_label(contenedor)
        ax.set_title(f"Frecuencia de {col}")
        fig.tight_layout()
        return fig

    def boxplot_por_grupo(self, col_num: str, col_grupo: str, df=None):
        datos = self._etiquetar(self.df if df is None else df)
        orden = sorted(datos[col_grupo].unique())
        fig, ax = self._figura()
        sns.boxplot(data=datos, x=col_grupo, y=col_num, hue=col_grupo, order=orden,
                    hue_order=orden, palette="Set2", legend=False, ax=ax)
        ax.set_title(f"{col_num} según {col_grupo}")
        fig.tight_layout()
        return fig

    def barras_apiladas(self, col_a: str, col_b: str, df=None):
        """Barras apiladas con proporciones (%) de col_b dentro de cada col_a."""
        datos = self._etiquetar(self.df if df is None else df)
        tabla = pd.crosstab(datos[col_a], datos[col_b], normalize="index") * 100
        fig, ax = self._figura()
        tabla.plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
        ax.set_ylabel("Porcentaje (%)")
        ax.set_title(f"{col_b} dentro de cada {col_a}")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title=col_b, loc="lower right", fontsize=8)
        fig.tight_layout()
        return fig

    def boxplot_escalas(self, cols: list):
        """Boxplots de varias variables en una misma escala, para compararlas."""
        largo = self.df[cols].melt(var_name="escala", value_name="valor")
        fig, ax = self._figura()
        sns.boxplot(data=largo, x="escala", y="valor", hue="escala", palette="Set2",
                    legend=False, ax=ax)
        ax.set_title("Comparación de escalas (1 a 10)")
        fig.tight_layout()
        return fig

    def interpretar_distribucion(self, col: str) -> str:
        """Texto breve sobre forma, concentración, asimetría y extremos."""
        s = self.df[col]
        asim = s.skew()
        if abs(asim) < 0.5:
            forma = "aproximadamente simétrica"
        elif asim > 0:
            forma = "con asimetría hacia la derecha (cola de valores altos)"
        else:
            forma = "con asimetría hacia la izquierda (cola de valores bajos)"
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        n_ext = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
        return (
            f"`{col}` tiene una distribución {forma} (asimetría = {asim:.2f}). "
            f"El 50% central de los datos está entre {q1:.2f} y {q3:.2f}; "
            f"la media es {s.mean():.2f} y la mediana {s.median():.2f}. "
            f"Valores extremos según IQR: {n_ext}."
        )

    def dispersion(self, x: str, y: str, df=None, hue: str | None = None, tendencia: bool = False):
        datos = self.df if df is None else df
        fig, ax = self._figura()
        sns.scatterplot(data=datos, x=x, y=y, hue=hue, alpha=0.6, ax=ax)
        if tendencia and len(datos) > 2:
            sns.regplot(data=datos, x=x, y=y, scatter=False, ax=ax, color="red",
                        line_kws={"linewidth": 2})
        ax.set_title(f"{y} vs {x}")
        fig.tight_layout()
        return fig

    def grafico_tramos(self, habito: str, bienestar: str, df=None, tramos: int = 4):
        """Promedio de una variable de bienestar por tramos (cuartiles) de un hábito."""
        datos = self.df if df is None else df
        try:
            grupos = pd.qcut(datos[habito], q=tramos, duplicates="drop")
        except ValueError:
            return None
        tabla = datos.groupby(grupos, observed=True)[bienestar].mean()
        if len(tabla) < 2:
            return None
        etiquetas = [f"{iv.left:.1f} - {iv.right:.1f}" for iv in tabla.index]
        fig, ax = self._figura()
        ax.bar(etiquetas, tabla.values, color="#3366FF")
        for i, v in enumerate(tabla.values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel(f"Tramos de {habito}")
        ax.set_ylabel(f"Promedio de {bienestar}")
        ax.set_title("Promedio por tramos")
        fig.tight_layout()
        return fig

    def barras_medias_por_grupo(self, cols: list, grupo: str = "depression_label"):
        """Barras agrupadas con la media de varias variables por grupo."""
        datos = self._etiquetar(self.df)
        medias = datos.groupby(grupo)[cols].mean().T
        fig, ax = self._figura(7, 4.5)
        medias.plot(kind="bar", ax=ax, color=["#9ECAE1", "#08519C"][:medias.shape[1]])
        ax.set_ylabel("Media")
        ax.set_title(f"Medias por grupo de {grupo}")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title=grupo, fontsize=8)
        fig.tight_layout()
        return fig

    def boxplot_simple(self, col: str, df=None):
        datos = self.df if df is None else df
        fig, ax = self._figura(6, 2.8)
        sns.boxplot(x=datos[col], ax=ax, color="#9ECAE1")
        ax.set_title(f"Boxplot de {col}")
        fig.tight_layout()
        return fig

    def heatmap_correlacion(self, df=None):
        datos = self.df if df is None else df
        corr = datos[self.columnas_numericas(incluir_etiqueta=True)].corr()
        fig, ax = self._figura(7, 5.5)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Matriz de correlación")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Módulo 1: Home
# ---------------------------------------------------------------------------
def mostrar_home():
    st.title("📊 Hábitos digitales y bienestar en adolescentes")
    st.subheader("Análisis Exploratorio de Datos (EDA) con Python y Streamlit")

    st.markdown(
        """
        **Objetivo del análisis:** identificar patrones exploratorios entre los
        hábitos digitales (redes sociales, plataforma, pantalla antes de dormir),
        el descanso, la actividad física, la interacción social y las variables de
        bienestar registradas en adolescentes de 13 a 19 años.

        > ⚠️ Este proyecto es **educativo y exploratorio**. Sus resultados no
        > constituyen un diagnóstico clínico ni sustituyen la valoración de
        > profesionales de la salud.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👩‍💻 Datos de la autora")
        st.markdown(
            f"""
            - **Nombre completo:** {AUTOR}
            - **Curso / Especialización:** {CURSO}
            - **Año:** {ANIO}
            """
        )
        st.markdown("### 🛠️ Tecnologías utilizadas")
        st.markdown(
            """
            - Python
            - Pandas y NumPy
            - Matplotlib y Seaborn
            - Streamlit
            - GitHub y Streamlit Community Cloud
            """
        )

    with col2:
        st.markdown("### 🗂️ Sobre el dataset")
        st.markdown(
            """
            El archivo `Teen_Mental_Health_Dataset.csv` contiene **1.200 registros
            y 13 variables**. Incluye uso diario de redes sociales, plataforma
            utilizada, horas de sueño, tiempo de pantalla antes de dormir,
            rendimiento académico, actividad física e interacción social.

            También incorpora escalas de estrés, ansiedad y nivel de dependencia
            (1 a 10), además de `depression_label`, una etiqueta binaria propia
            del dataset (0 = ausencia, 1 = presencia de la condición etiquetada).
            """
        )

    st.info("👈 Usa el menú lateral para navegar. Primero carga el dataset en el módulo 2.")


# ---------------------------------------------------------------------------
# Módulo 2: Carga del dataset
# ---------------------------------------------------------------------------
def mostrar_carga():
    st.title("📂 Carga del dataset")
    st.write("Sube el archivo **Teen_Mental_Health_Dataset.csv** para habilitar el análisis.")

    archivo = st.file_uploader("Selecciona el archivo CSV", type=["csv"])

    if archivo is None:
        st.warning("Aún no se ha cargado ningún archivo. El EDA permanecerá bloqueado.")
        st.session_state.pop("analyzer", None)
        return

    es_valido, mensaje, df = DataAnalyzer.validar_archivo(archivo)

    if not es_valido:
        st.error(mensaje)
        st.session_state.pop("analyzer", None)
        return

    st.success(mensaje)
    st.session_state["analyzer"] = DataAnalyzer(df)
    analyzer = st.session_state["analyzer"]

    # Dimensiones
    c1, c2 = st.columns(2)
    c1.metric("Filas", f"{analyzer.filas:,}")
    c2.metric("Columnas", analyzer.columnas)

    # Vista previa (head)
    st.markdown("### Vista previa del dataset")
    n = st.slider("Número de filas a mostrar", min_value=5, max_value=50, value=5, step=5)
    st.dataframe(analyzer.df.head(n), use_container_width=True)


# ---------------------------------------------------------------------------
# Módulo 3: EDA (10 ítems organizados en tabs)
# ---------------------------------------------------------------------------
def item_1(a: DataAnalyzer):
    st.header("Ítem 1: Información general del dataset")
    st.write(
        "Antes de analizar, conviene conocer la estructura del archivo: cuántas "
        "filas y columnas tiene, qué tipo de dato tiene cada variable y si existen "
        "valores nulos o registros repetidos."
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Filas", f"{a.filas:,}")
    m2.metric("Columnas", a.columnas)
    m3.metric("Valores nulos (total)", int(a.valores_nulos()["nulos"].sum()))
    m4.metric("Registros duplicados", a.num_duplicados())

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Salida de .info()")
        st.code(a.info_texto())
    with col_der:
        st.subheader("Tipos de datos y valores únicos")
        st.dataframe(a.resumen_tipos(), use_container_width=True)

    if a.num_duplicados() == 0 and a.valores_nulos()["nulos"].sum() == 0:
        st.success("El dataset está completo: sin nulos y sin duplicados. "
                   "Se puede avanzar directamente al análisis de distribuciones.")
    else:
        st.warning("Se detectaron nulos o duplicados; conviene revisarlos antes de continuar.")


def item_2(a: DataAnalyzer):
    st.header("Ítem 2: Clasificación de variables")
    st.write(
        "Se usa la función personalizada `clasificar_variables()` de la clase "
        "`DataAnalyzer`, que separa las variables según su tipo de dato."
    )

    clases = a.clasificar_variables()
    c1, c2, c3 = st.columns(3)
    c1.metric("Numéricas", len(clases["numericas"]))
    c2.metric("Categóricas", len(clases["categoricas"]))
    c3.metric("Binarias", len(clases["binarias"]))

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Variables numéricas")
        for v in clases["numericas"]:
            st.write(f"- `{v}`")
    with col_der:
        st.subheader("Variables categóricas")
        for v in clases["categoricas"]:
            st.write(f"- `{v}` ({a.df[v].nunique()} categorías)")

    st.info(
        "Nota: `depression_label` está codificada con 0 y 1, por eso es numérica "
        "por tipo de dato, pero en la práctica funciona como una etiqueta binaria "
        "(categoría). Se tratará con ese criterio en las comparaciones entre grupos."
    )

    if st.checkbox("Ver gráfico del conteo por tipo de variable", key="chk_item2"):
        fig, ax = plt.subplots(figsize=(5, 3))
        etiquetas = ["Numéricas", "Categóricas"]
        valores = [len(clases["numericas"]), len(clases["categoricas"])]
        ax.bar(etiquetas, valores, color=["#3366FF", "#9ECAE1"])
        for i, v in enumerate(valores):
            ax.text(i, v + 0.1, str(v), ha="center")
        ax.set_ylabel("Cantidad de variables")
        fig.tight_layout()
        mostrar_fig(fig)


def item_3(a: DataAnalyzer):
    st.header("Ítem 3: Estadísticas descriptivas")
    st.write(
        "Se resumen las variables numéricas con `.describe()`, ampliado con "
        "mediana, moda, rango intercuartílico (IQR) y coeficiente de variación (CV)."
    )

    st.dataframe(a.estadisticas_descriptivas(), use_container_width=True)

    st.subheader("Mira una variable en detalle")
    variable = st.selectbox("Variable numérica", a.columnas_numericas(), key="sel_item3")
    serie = a.df[variable]
    media, mediana, desv = serie.mean(), serie.median(), serie.std()

    col_izq, col_der = st.columns([1, 2])
    with col_izq:
        st.metric("Media", f"{media:.2f}")
        st.metric("Mediana", f"{mediana:.2f}")
        st.metric("Desv. estándar", f"{desv:.2f}")
        st.metric("Q1 - Q3", f"{serie.quantile(0.25):.2f} - {serie.quantile(0.75):.2f}")
    with col_der:
        mostrar_fig(a.boxplot_simple(variable))

    # Interpretación automática básica (media vs mediana)
    diferencia = abs(media - mediana) / desv if desv else 0
    if diferencia < 0.1:
        forma = "es aproximadamente simétrica (media y mediana son muy cercanas)"
    elif media > mediana:
        forma = "tiene una ligera asimetría hacia valores altos (media mayor que la mediana)"
    else:
        forma = "tiene una ligera asimetría hacia valores bajos (media menor que la mediana)"
    st.write(f"**Interpretación:** la variable `{variable}` {forma}.")

    st.subheader("Detección preliminar de valores extremos (regla 1.5 × IQR)")
    st.dataframe(a.outliers_iqr(), use_container_width=True)
    if a.outliers_iqr()["n_extremos"].sum() == 0:
        st.success("Ninguna variable presenta valores extremos según la regla del IQR.")
    else:
        st.warning("Hay variables con posibles valores extremos; conviene revisarlas.")


def item_4(a: DataAnalyzer):
    st.header("Ítem 4: Análisis de valores faltantes")
    st.write("Se cuantifican los valores faltantes por variable, en conteo y en porcentaje.")

    nulos = a.valores_nulos()
    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Conteo y porcentaje")
        st.dataframe(nulos, use_container_width=True)
    with col_der:
        st.subheader("Visualización")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.barh(nulos.index, nulos["nulos"], color="#3366FF")
        ax.set_xlabel("Cantidad de nulos")
        ax.set_xlim(0, max(1, nulos["nulos"].max()))
        fig.tight_layout()
        mostrar_fig(fig)

    if nulos["nulos"].sum() == 0:
        st.success("No hay valores faltantes en ninguna variable.")

    st.subheader("¿Qué hacer con los valores faltantes?")
    st.write(
        "En este dataset no es necesario tratarlos. En otros casos, la decisión "
        "depende de la cantidad y de la variable afectada:"
    )
    st.markdown(
        """
        - **Conservarlos** si son pocos y no distorsionan el análisis.
        - **Eliminar filas** cuando el porcentaje es muy bajo.
        - **Imputar** con la media o la mediana (numéricas) o la moda (categóricas).
        - **Revisar la causa** antes de decidir, para no introducir sesgos.
        """
    )


def item_5(a: DataAnalyzer):
    st.header("Ítem 5: Distribución de variables numéricas")
    st.write(
        "Los histogramas muestran cómo se reparten los valores de cada variable: "
        "su forma, dónde se concentran, si son simétricos y si hay valores extremos. "
        "La línea roja marca la media y la verde la mediana."
    )

    col_izq, col_der = st.columns([1, 2])
    with col_izq:
        variable = st.selectbox("Variable numérica", a.columnas_numericas(), key="sel_item5")
        bins = st.slider("Número de intervalos (bins)", 5, 50, 20, key="sld_item5")
    with col_der:
        mostrar_fig(a.histograma(variable, bins=bins))
    st.write(f"**Interpretación:** {a.interpretar_distribucion(variable)}")

    if st.checkbox("Ver histogramas de todas las variables numéricas", key="chk_item5"):
        cols = a.columnas_numericas()
        for i in range(0, len(cols), 3):
            fila = st.columns(3)
            for caja, var in zip(fila, cols[i:i + 3]):
                with caja:
                    mostrar_fig(a.histograma(var, bins=bins))

    st.subheader("Comparación de escalas: estrés, ansiedad y dependencia")
    escalas = ["stress_level", "anxiety_level", "addiction_level"]
    resumen = a.df[escalas].agg(["mean", "median", "std", "min", "max", "skew"]).T.round(2)
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(resumen, use_container_width=True)
    with c2:
        mostrar_fig(a.boxplot_escalas(escalas))

    mayor = resumen["mean"].idxmax()
    st.write(
        f"**Interpretación:** las tres variables están medidas en una escala de 1 a 10. "
        f"La media más alta corresponde a `{mayor}` ({resumen.loc[mayor, 'mean']:.2f}), "
        f"y la más baja a `{resumen['mean'].idxmin()}` ({resumen['mean'].min():.2f})."
    )
    st.info(
        "Estas escalas se usan solo para comparar distribuciones dentro del dataset. "
        "No constituyen un diagnóstico clínico."
    )


def item_6(a: DataAnalyzer):
    st.header("Ítem 6: Análisis de variables categóricas")
    st.write(
        "Se observan las frecuencias y proporciones de las variables categóricas. "
        "Se incluye `depression_label`, que funciona como etiqueta binaria."
    )

    opciones = a.columnas_categoricas() + ["depression_label"]
    variable = st.selectbox("Variable categórica", opciones, key="sel_item6")
    frec = a.frecuencias(variable)

    col_izq, col_der = st.columns(2)
    with col_izq:
        st.subheader("Conteos y proporciones")
        st.dataframe(frec, use_container_width=True)
        st.write(
            f"Categoría más frecuente (moda): **{frec.index[0]}** "
            f"({frec['proporcion_%'].iloc[0]}% de los registros)."
        )
    with col_der:
        mostrar_fig(a.grafico_barras(variable))

    st.subheader("Categorías relevantes para el caso")
    n1 = int((a.df["depression_label"] == 1).sum())
    m1, m2, m3 = st.columns(3)
    m1.metric("Registros con etiqueta 1", n1, f"{n1 / a.filas * 100:.1f}% del total")
    m2.metric("Plataforma más frecuente", a.frecuencias("platform_usage").index[0])
    m3.metric("Interacción social más frecuente", a.frecuencias("social_interaction_level").index[0])

    st.warning(
        f"`depression_label` está muy desbalanceada: solo {n1} de {a.filas:,} registros "
        f"({n1 / a.filas * 100:.1f}%) tienen la etiqueta 1. Las comparaciones con este grupo "
        "deben interpretarse con prudencia."
    )


def _texto_diferencia(a: DataAnalyzer, var: str) -> str:
    medias = a.df.groupby("depression_label")[var].mean()
    if 0 not in medias.index or 1 not in medias.index:
        return "No hay datos suficientes en ambos grupos para comparar."
    dif = medias[1] - medias[0]
    sentido = "más" if dif > 0 else "menos"
    return (
        f"En promedio, el grupo con etiqueta 1 registra {abs(dif):.2f} unidades {sentido} de "
        f"`{var}` que el grupo con etiqueta 0 ({medias[1]:.2f} frente a {medias[0]:.2f})."
    )


def item_7(a: DataAnalyzer):
    st.header("Ítem 7: Análisis bivariado (numérico vs categórico)")
    st.write(
        "Se comparan variables numéricas entre los dos grupos de `depression_label` "
        "usando boxplots y estadísticos por grupo."
    )

    n0 = int((a.df["depression_label"] == 0).sum())
    n1 = int((a.df["depression_label"] == 1).sum())
    st.warning(
        f"Tamaño de los grupos: etiqueta 0 = {n0} registros; etiqueta 1 = {n1} registros. "
        "Al ser grupos tan desiguales, las diferencias son exploratorias y no concluyentes."
    )

    st.subheader("Comparaciones principales")
    c1, c2 = st.columns(2)
    for caja, var in zip((c1, c2), ["daily_social_media_hours", "sleep_hours"]):
        with caja:
            st.markdown(f"**{var}**")
            mostrar_fig(a.boxplot_por_grupo(var, "depression_label"))
            st.dataframe(a.comparar_grupos(var, "depression_label"), use_container_width=True)
            st.write(_texto_diferencia(a, var))

    st.subheader("Explora otra variable")
    otras = ["academic_performance", "physical_activity", "screen_time_before_sleep",
             "stress_level", "anxiety_level", "addiction_level"]
    var = st.selectbox("Variable numérica", otras, key="sel_item7")
    izq, der = st.columns(2)
    with izq:
        st.dataframe(a.comparar_grupos(var, "depression_label"), use_container_width=True)
        st.write(_texto_diferencia(a, var))
    with der:
        mostrar_fig(a.boxplot_por_grupo(var, "depression_label"))


def item_8(a: DataAnalyzer):
    st.header("Ítem 8: Análisis bivariado (categórico vs categórico)")
    st.write(
        "Se cruzan dos variables categóricas y se muestran los porcentajes por fila, "
        "es decir, cómo se reparte la segunda variable dentro de cada categoría de la primera."
    )

    pares = {
        "platform_usage vs depression_label": ("platform_usage", "depression_label"),
        "social_interaction_level vs depression_label": ("social_interaction_level", "depression_label"),
        "gender vs platform_usage": ("gender", "platform_usage"),
    }
    eleccion = st.selectbox("Comparación", list(pares), key="sel_item8")
    col_a, col_b = pares[eleccion]

    izq, der = st.columns(2)
    with izq:
        st.subheader("Porcentaje por fila (%)")
        tabla = a.tabla_cruzada(col_a, col_b)
        st.dataframe(tabla, use_container_width=True)
        if st.checkbox("Mostrar conteos absolutos", key="chk_item8"):
            st.dataframe(a.tabla_cruzada(col_a, col_b, porcentaje=False), use_container_width=True)
    with der:
        mostrar_fig(a.barras_apiladas(col_a, col_b))

    if col_b == "depression_label":
        col_mayor = "1 - Presencia"
    else:
        col_mayor = (tabla.max() - tabla.min()).idxmax()
    st.write(
        f"**Interpretación:** en `{col_b} = {col_mayor}`, el porcentaje va de "
        f"{tabla[col_mayor].min():.1f}% (en {tabla[col_mayor].idxmin()}) a "
        f"{tabla[col_mayor].max():.1f}% (en {tabla[col_mayor].idxmax()})."
    )
    st.caption(
        "Diferencias de pocos puntos porcentuales, sobre todo con la etiqueta 1 que tiene "
        "pocos casos, no permiten conclusiones firmes. Es un análisis exploratorio."
    )


def _texto_correlacion(r: float) -> str:
    if np.isnan(r):
        return "No se puede calcular la correlación con los datos filtrados."
    if abs(r) < 0.1:
        return f"La correlación de Pearson es r = {r:.2f}: relación lineal muy débil o nula."
    fuerza = "débil" if abs(r) < 0.3 else "moderada" if abs(r) < 0.5 else "fuerte"
    sentido = "positiva" if r > 0 else "negativa"
    return f"La correlación de Pearson es r = {r:.2f}: relación lineal {fuerza} y {sentido}."


def item_9(a: DataAnalyzer):
    st.header("Ítem 9: Análisis basado en parámetros seleccionados")
    st.write(
        "Filtra a los adolescentes por edad, género, plataforma e interacción social, "
        "y elige una variable de bienestar y una de hábitos digitales para analizarlas."
    )

    st.subheader("1. Filtros")
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        edad_min, edad_max = int(a.df["age"].min()), int(a.df["age"].max())
        edad = st.slider("Rango de edad", edad_min, edad_max, (edad_min, edad_max), key="sld_item9")
    with f2:
        generos_op = sorted(a.df["gender"].unique())
        generos = st.multiselect("Género", generos_op, default=generos_op, key="ms_gen_item9")
    with f3:
        plat_op = sorted(a.df["platform_usage"].unique())
        plataformas = st.multiselect("Plataforma", plat_op, default=plat_op, key="ms_plat_item9")
    with f4:
        orden = ["low", "medium", "high"]
        inter_op = [x for x in orden if x in a.df["social_interaction_level"].unique()]
        interacciones = st.multiselect("Interacción social", inter_op, default=inter_op, key="ms_int_item9")

    if not (generos and plataformas and interacciones):
        st.warning("Selecciona al menos una opción en cada filtro (género, plataforma e interacción).")
        return

    filtrado = a.filtrar(edad=edad, generos=generos, plataformas=plataformas, interacciones=interacciones)
    if filtrado.empty:
        st.error("No hay registros con esa combinación de filtros. Amplía la selección.")
        return

    st.subheader("2. Variables a analizar")
    v1, v2, v3 = st.columns(3)
    with v1:
        bienestar = st.selectbox(
            "Variable de bienestar o descanso",
            ["stress_level", "anxiety_level", "addiction_level", "sleep_hours"], key="sel_bien_item9")
    with v2:
        habito = st.selectbox(
            "Variable de hábitos digitales",
            ["daily_social_media_hours", "screen_time_before_sleep"], key="sel_hab_item9")
    with v3:
        color = st.selectbox(
            "Colorear puntos por",
            ["Ninguno", "gender", "platform_usage", "social_interaction_level"], key="sel_hue_item9")

    tendencia = st.checkbox("Mostrar línea de tendencia", value=True, key="chk_tend_item9")

    st.subheader("3. Resultados")
    r = filtrado[habito].corr(filtrado[bienestar])
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Registros filtrados", f"{len(filtrado):,}", f"{len(filtrado) / a.filas * 100:.1f}% del total")
    m2.metric(f"Media de {habito}", f"{filtrado[habito].mean():.2f}")
    m3.metric(f"Media de {bienestar}", f"{filtrado[bienestar].mean():.2f}")
    m4.metric("Correlación (r)", "N/D" if np.isnan(r) else f"{r:.2f}")

    if len(filtrado) < 30:
        st.warning("Hay pocos registros con estos filtros; los resultados son poco estables.")

    izq, der = st.columns(2)
    with izq:
        hue = None if color == "Ninguno" else color
        mostrar_fig(a.dispersion(habito, bienestar, df=filtrado, hue=hue, tendencia=tendencia))
    with der:
        fig = a.grafico_tramos(habito, bienestar, df=filtrado)
        if fig is None:
            st.info("No hay suficiente variación en los datos filtrados para armar tramos.")
        else:
            mostrar_fig(fig)

    st.write(f"**Interpretación:** {_texto_correlacion(r)} Una correlación no implica causalidad.")

    if st.checkbox("Mostrar tabla de datos filtrados", key="chk_tabla_item9"):
        st.dataframe(filtrado, use_container_width=True)


def item_10(a: DataAnalyzer):
    st.header("Ítem 10: Hallazgos clave")
    st.write(
        "Resumen visual y principales ideas surgidas del análisis. Todo es exploratorio: "
        "describe patrones de este dataset y no predice ni diagnostica."
    )

    if not {0, 1}.issubset(set(a.df["depression_label"].unique())):
        st.info("Este resumen requiere que `depression_label` tenga las categorías 0 y 1.")
        return

    st.subheader("Visualización resumen")
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.barras_medias_por_grupo(
            ["daily_social_media_hours", "sleep_hours", "stress_level", "anxiety_level"]))
    with der:
        mostrar_fig(a.heatmap_correlacion())

    # Cifras para los insights
    medias = a.df.groupby("depression_label").mean(numeric_only=True)
    n1 = int((a.df["depression_label"] == 1).sum())
    habitos = ["daily_social_media_hours", "screen_time_before_sleep"]
    escalas = ["stress_level", "anxiety_level", "addiction_level", "sleep_hours"]
    max_corr = a.df[habitos + escalas].corr().loc[habitos, escalas].abs().to_numpy().max()

    st.subheader("Insights principales")
    st.markdown(
        f"""
        1. **Datos completos:** {a.filas:,} registros, {int(a.valores_nulos()['nulos'].sum())} nulos y
           {a.num_duplicados()} duplicados, por lo que no fue necesario limpiar.
        2. **Etiqueta desbalanceada:** solo {n1} registros ({n1 / a.filas * 100:.1f}%) tienen
           `depression_label = 1`. Las comparaciones con ese grupo son orientativas.
        3. **Uso de redes:** el grupo con etiqueta 1 usa en promedio
           {medias.loc[1, 'daily_social_media_hours']:.2f} h diarias frente a
           {medias.loc[0, 'daily_social_media_hours']:.2f} h del grupo 0.
        4. **Descanso:** el grupo con etiqueta 1 duerme en promedio
           {medias.loc[1, 'sleep_hours']:.2f} h frente a {medias.loc[0, 'sleep_hours']:.2f} h.
        5. **Estrés y ansiedad:** en el grupo con etiqueta 1 las medias son
           {medias.loc[1, 'stress_level']:.2f} y {medias.loc[1, 'anxiety_level']:.2f}, frente a
           {medias.loc[0, 'stress_level']:.2f} y {medias.loc[0, 'anxiety_level']:.2f} en el grupo 0.
           En cambio, `addiction_level` y `screen_time_before_sleep` casi no cambian entre grupos.
        6. **Sin relación lineal general:** en todo el dataset, la correlación absoluta máxima entre
           los hábitos digitales y las escalas de bienestar es {max_corr:.2f}, es decir, muy débil.
        7. **Variables sin diferencias marcadas:** plataforma, nivel de interacción social y género
           reparten la etiqueta de forma parecida (revisar Ítem 8).
        """
    )

    st.subheader("Recomendaciones de interpretación")
    st.markdown(
        """
        - Considerar el **sueño** y las **horas de uso de redes** como variables a observar
          primero, porque son donde más se separan los grupos.
        - No asumir que más horas en una plataforma específica implica mayor o menor bienestar:
          en estos datos la plataforma no muestra diferencias claras.
        - Antes de sacar conclusiones firmes, contar con **más casos** en el grupo con etiqueta 1.
        - Usar estos resultados para **orientar preguntas y acciones educativas**, no para
          clasificar a personas. No sustituyen la valoración de profesionales de la salud.
        """
    )


def mostrar_eda():
    st.title("🔎 Análisis Exploratorio de Datos")
    analyzer = st.session_state.get("analyzer")

    if analyzer is None:
        st.error("⛔ Primero debes cargar el dataset en el módulo 'Carga del dataset'.")
        return

    nombres = [
        "1. Info general", "2. Variables", "3. Estadísticas", "4. Faltantes",
        "5. Distribuciones", "6. Categóricas", "7. Num vs Cat", "8. Cat vs Cat",
        "9. Filtros", "10. Hallazgos",
    ]
    tabs = st.tabs(nombres)

    with tabs[0]:
        item_1(analyzer)
    with tabs[1]:
        item_2(analyzer)
    with tabs[2]:
        item_3(analyzer)
    with tabs[3]:
        item_4(analyzer)
    with tabs[4]:
        item_5(analyzer)
    with tabs[5]:
        item_6(analyzer)
    with tabs[6]:
        item_7(analyzer)
    with tabs[7]:
        item_8(analyzer)
    with tabs[8]:
        item_9(analyzer)
    with tabs[9]:
        item_10(analyzer)


# ---------------------------------------------------------------------------
# Módulo 4: Conclusiones (5 conclusiones con evidencia)
# ---------------------------------------------------------------------------
def mostrar_conclusiones():
    st.title("✅ Conclusiones finales")
    a = st.session_state.get("analyzer")
    if a is None:
        st.error("⛔ Primero debes cargar el dataset en el módulo 'Carga del dataset'.")
        return

    if not {0, 1}.issubset(set(a.df["depression_label"].unique())):
        st.info("Las conclusiones requieren que `depression_label` tenga las categorías 0 y 1.")
        return

    st.caption(
        "Conclusiones basadas en el análisis exploratorio. Describen patrones de este dataset, "
        "no predicen ni diagnostican, y sirven como apoyo para orientar decisiones educativas."
    )

    medias = a.df.groupby("depression_label").mean(numeric_only=True)
    n0 = int((a.df["depression_label"] == 0).sum())
    n1 = int((a.df["depression_label"] == 1).sum())

    # ---------------- Conclusión 1 ----------------
    st.header("Conclusión 1: el uso de redes sociales es mayor en el grupo con etiqueta 1")
    st.markdown(
        f"El grupo con `depression_label = 1` registra en promedio "
        f"**{medias.loc[1, 'daily_social_media_hours']:.2f} h diarias** de redes sociales, frente a "
        f"**{medias.loc[0, 'daily_social_media_hours']:.2f} h** del grupo 0. "
        "**Sugerencia:** incluir las horas de uso como una de las variables a observar "
        "en acciones de sensibilización sobre hábitos digitales."
    )
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.boxplot_por_grupo("daily_social_media_hours", "depression_label"))
    with der:
        st.dataframe(a.comparar_grupos("daily_social_media_hours", "depression_label"),
                     use_container_width=True)
    st.caption("Evidencia: Ítem 7, boxplot y tabla de estadísticos por grupo.")

    # ---------------- Conclusión 2 ----------------
    st.header("Conclusión 2: el grupo con etiqueta 1 duerme menos")
    st.markdown(
        f"El promedio de sueño es de **{medias.loc[1, 'sleep_hours']:.2f} h** en el grupo con "
        f"etiqueta 1 y de **{medias.loc[0, 'sleep_hours']:.2f} h** en el grupo 0, una diferencia de "
        f"{medias.loc[0, 'sleep_hours'] - medias.loc[1, 'sleep_hours']:.2f} h. "
        "**Sugerencia:** considerar la higiene del sueño como un tema prioritario "
        "en programas de bienestar para adolescentes."
    )
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.boxplot_por_grupo("sleep_hours", "depression_label"))
    with der:
        st.dataframe(a.comparar_grupos("sleep_hours", "depression_label"), use_container_width=True)
    st.caption("Evidencia: Ítem 7, boxplot y tabla de estadísticos por grupo.")

    # ---------------- Conclusión 3 ----------------
    st.header("Conclusión 3: estrés y ansiedad se separan entre grupos, pero dependencia y pantalla nocturna no")
    st.markdown(
        f"En el grupo con etiqueta 1, el estrés promedia **{medias.loc[1, 'stress_level']:.2f}** y la "
        f"ansiedad **{medias.loc[1, 'anxiety_level']:.2f}** (escala de 1 a 10), frente a "
        f"**{medias.loc[0, 'stress_level']:.2f}** y **{medias.loc[0, 'anxiety_level']:.2f}** en el grupo 0. "
        f"En cambio, `addiction_level` ({medias.loc[1, 'addiction_level']:.2f} frente a "
        f"{medias.loc[0, 'addiction_level']:.2f}) y `screen_time_before_sleep` "
        f"({medias.loc[1, 'screen_time_before_sleep']:.2f} h frente a "
        f"{medias.loc[0, 'screen_time_before_sleep']:.2f} h) casi no cambian. "
        "**Sugerencia:** no asumir que un mayor nivel de dependencia o más pantalla antes de "
        "dormir distingue por sí solo a los grupos; mirar el conjunto de variables."
    )
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.barras_medias_por_grupo(
            ["stress_level", "anxiety_level", "addiction_level", "screen_time_before_sleep"]))
    with der:
        st.dataframe(
            medias[["stress_level", "anxiety_level", "addiction_level",
                    "screen_time_before_sleep"]].round(2).T,
            use_container_width=True)
    st.caption("Evidencia: Ítems 5 y 10, medias por grupo. Las escalas no son un diagnóstico clínico.")

    # ---------------- Conclusión 4 ----------------
    st.header("Conclusión 4: plataforma, interacción social, rendimiento y actividad física muestran poca diferencia")
    tab_plat = a.tabla_cruzada("platform_usage", "depression_label")["1 - Presencia"]
    tab_int = a.tabla_cruzada("social_interaction_level", "depression_label")["1 - Presencia"]
    st.markdown(
        f"El porcentaje con etiqueta 1 va de **{tab_plat.min():.1f}% a {tab_plat.max():.1f}%** según la "
        f"plataforma y de **{tab_int.min():.1f}% a {tab_int.max():.1f}%** según el nivel de interacción "
        f"social. El rendimiento académico ({medias.loc[1, 'academic_performance']:.2f} frente a "
        f"{medias.loc[0, 'academic_performance']:.2f}) y la actividad física "
        f"({medias.loc[1, 'physical_activity']:.2f} h frente a {medias.loc[0, 'physical_activity']:.2f} h) "
        "también son muy parecidos. **Sugerencia:** evitar enfocar las acciones solo en una "
        "plataforma o en un nivel de interacción social concreto."
    )
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.barras_apiladas("platform_usage", "depression_label"))
    with der:
        st.write("Porcentaje con etiqueta 1 por nivel de interacción social")
        st.dataframe(a.tabla_cruzada("social_interaction_level", "depression_label"),
                     use_container_width=True)
        st.write("Medias de rendimiento académico y actividad física por grupo")
        st.dataframe(medias[["academic_performance", "physical_activity"]].round(2),
                     use_container_width=True)
    st.caption("Evidencia: Ítem 8, tablas cruzadas y barras apiladas; Ítem 7, medias por grupo.")

    # ---------------- Conclusión 5 ----------------
    habitos = ["daily_social_media_hours", "screen_time_before_sleep"]
    escalas = ["stress_level", "anxiety_level", "addiction_level", "sleep_hours"]
    max_corr = a.df[habitos + escalas].corr().loc[habitos, escalas].abs().to_numpy().max()
    st.header("Conclusión 5: los hallazgos son exploratorios por el desbalance de la etiqueta")
    st.markdown(
        f"Solo **{n1} registros ({n1 / a.filas * 100:.1f}%)** tienen etiqueta 1, frente a {n0:,} con "
        f"etiqueta 0. Además, en todo el dataset la correlación absoluta máxima entre hábitos digitales "
        f"y escalas de bienestar es de apenas **{max_corr:.2f}**. Los patrones observados describen "
        "diferencias entre grupos, no una relación gradual ni una causa. **Sugerencia:** "
        "reunir más casos antes de tomar decisiones de fondo y usar estos resultados solo con fines "
        "educativos y de orientación, sin sustituir la valoración de profesionales de la salud."
    )
    izq, der = st.columns(2)
    with izq:
        mostrar_fig(a.grafico_barras("depression_label"))
    with der:
        st.dataframe(a.frecuencias("depression_label"), use_container_width=True)
    st.caption("Evidencia: Ítem 6, frecuencias de la etiqueta; Ítem 10, matriz de correlación.")


# ---------------------------------------------------------------------------
# Navegación principal (sidebar)
# ---------------------------------------------------------------------------
def main():
    st.sidebar.title("🧭 Navegación")
    modulo = st.sidebar.radio(
        "Ir a:",
        ["🏠 Home", "📂 Carga del dataset", "🔎 EDA", "✅ Conclusiones"],
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"{AUTOR}\n\n{CURSO} · {ANIO}")

    if modulo == "🏠 Home":
        mostrar_home()
    elif modulo == "📂 Carga del dataset":
        mostrar_carga()
    elif modulo == "🔎 EDA":
        mostrar_eda()
    else:
        mostrar_conclusiones()


if __name__ == "__main__":
    main()
