# 📊 Hábitos digitales y bienestar en adolescentes: EDA con Python y Streamlit

Aplicación interactiva de **Análisis Exploratorio de Datos (EDA)** construida con Python y Streamlit
a partir del dataset `Teen_Mental_Health_Dataset.csv`.

> ⚠️ Proyecto **educativo y exploratorio**. Los resultados no constituyen un diagnóstico clínico
> ni sustituyen la valoración de profesionales de la salud.

---

## 👩‍💻 Autora

- **Nombre:** Elizabeth Fiorella Cotrina Gallardo
- **Curso:** Especialización en Python for Analytics (DMC Institute)
- **Año:** 2026

## 🔗 Links relevantes

- 🌐 **Aplicación desplegada:** https://teen-mental-health-eda-2v4jparytpz3btxtztej2n.streamlit.app/
- 💻 **Repositorio GitHub:** https://github.com/elyficg-ops/teen-mental-health-eda

## 🎯 Descripción del proyecto

El objetivo **no** es construir modelos predictivos, sino analizar, limpiar, transformar y visualizar
los datos para identificar patrones exploratorios entre hábitos digitales, descanso, actividad física,
interacción social y variables de bienestar en adolescentes de 13 a 19 años.

La aplicación se organiza en módulos navegables desde un menú lateral:

1. **Home:** presentación del proyecto.
2. **Carga del dataset:** subida del CSV con validación, vista previa y dimensiones.
3. **EDA:** 10 ítems de análisis organizados en pestañas.
4. **Conclusiones:** 5 conclusiones vinculadas a evidencia visual o estadística.

### Ítems del EDA

| # | Ítem |
|---|---|
| 1 | Información general del dataset |
| 2 | Clasificación de variables |
| 3 | Estadísticas descriptivas y valores extremos |
| 4 | Análisis de valores faltantes |
| 5 | Distribución de variables numéricas |
| 6 | Análisis de variables categóricas |
| 7 | Análisis bivariado: numérico vs categórico |
| 8 | Análisis bivariado: categórico vs categórico |
| 9 | Análisis con filtros y parámetros seleccionados |
| 10 | Hallazgos clave |

## 🖼️ Capturas de la aplicación

<!-- Guarda tus capturas en la carpeta images/ con estos nombres, o cámbialos aquí -->

| Home | Carga del dataset |
|---|---|
| ![Home](images/home.png) | ![Carga](images/carga.png) |

| EDA | Conclusiones |
|---|---|
| ![EDA](images/eda.png) | ![Conclusiones](images/conclusiones.png) |

## 🛠️ Tecnologías utilizadas

- Python
- Pandas y NumPy
- Matplotlib y Seaborn
- Streamlit
- Programación Orientada a Objetos (clase `DataAnalyzer`)

## ▶️ Instrucciones de ejecución

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/elyficg-ops/teen-mental-health-eda.git
   cd teen-mental-health-eda
   ```
2. (Opcional) Crear y activar un entorno virtual:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
4. Ejecutar la aplicación:
   ```bash
   python -m streamlit run app.py
   ```
5. En el módulo **Carga del dataset**, subir el archivo `data/Teen_Mental_Health_Dataset.csv`.

## 📁 Estructura del repositorio

```
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   └── Teen_Mental_Health_Dataset.csv
└── images/
```

## 🗂️ Variables principales

El dataset contiene **1.200 registros y 13 variables**, sin valores faltantes ni duplicados.

| Variable | Descripción |
|---|---|
| `age` | Edad del adolescente (13 a 19 años) |
| `gender` | Género registrado |
| `daily_social_media_hours` | Horas diarias de uso de redes sociales |
| `platform_usage` | Plataforma utilizada: Instagram, TikTok o ambas |
| `sleep_hours` | Horas de sueño por día |
| `screen_time_before_sleep` | Horas de pantalla antes de dormir |
| `academic_performance` | Indicador de rendimiento académico |
| `physical_activity` | Horas de actividad física |
| `social_interaction_level` | Nivel de interacción social: bajo, medio o alto |
| `stress_level` | Nivel de estrés (escala 1 a 10) |
| `anxiety_level` | Nivel de ansiedad (escala 1 a 10) |
| `addiction_level` | Nivel de dependencia o uso problemático (escala 1 a 10) |
| `depression_label` | Etiqueta binaria del dataset: 0 = ausencia, 1 = presencia de la condición etiquetada |

## 🔎 Hallazgos destacados

- El grupo con `depression_label = 1` usa más horas de redes sociales y duerme menos en promedio.
- Esa etiqueta está muy desbalanceada (31 de 1.200 registros), por lo que los resultados son orientativos.
- Plataforma, interacción social, rendimiento académico y actividad física muestran poca diferencia entre grupos.

---

*Proyecto desarrollado como parte de la Especialización en Python for Analytics.*
