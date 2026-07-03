"""
RiskPrevention — Herramienta inteligente de predicción de riesgo de deserción laboral
Curso: Sistemas Inteligentes y Machine Learning — Universidad Privada del Norte
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── CONFIGURACIÓN DE PÁGINA ──────────────────────────────────────────────────
st.set_page_config(
    page_title="RiskPrevention",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── ESTILOS CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header principal */
    .rp-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .rp-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .rp-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.85; }

    /* Tarjetas de métricas */
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid #1565c0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card h4 { margin: 0; color: #555; font-size: 0.8rem; text-transform: uppercase; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 1.5rem; font-weight: 700; color: #1a237e; }

    /* Tarjeta de riesgo */
    .risk-bajo  { background:#e8f5e9; border:2px solid #43a047; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-medio { background:#fff8e1; border:2px solid #fb8c00; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-alto  { background:#ffebee; border:2px solid #e53935; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-label { font-size:2rem; font-weight:800; margin:0; }
    .risk-sub   { font-size:0.95rem; margin:0.3rem 0 0; }
    .bajo-text  { color:#2e7d32; }
    .medio-text { color:#e65100; }
    .alto-text  { color:#b71c1c; }

    /* Barra de probabilidad */
    .prob-bar-bg { background:#e0e0e0; border-radius:8px; height:20px; margin:0.5rem 0; }
    .prob-bar-fill { height:20px; border-radius:8px; transition: width 0.5s; }

    /* Recomendaciones */
    .rec-box {
        background: #f3f4f6;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-top: 1rem;
        border-left: 4px solid #1565c0;
    }
    .rec-box h4 { margin:0 0 0.6rem; color:#1a237e; }
    .rec-box ul { margin:0; padding-left:1.2rem; }
    .rec-box li { margin-bottom:0.4rem; font-size:0.95rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #f0f4ff; }

    /* Footer */
    .footer { text-align:center; color:#999; font-size:0.78rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ─── CARGA DE ARTEFACTOS ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo de predicción...")
def cargar_modelo():
    archivos = ["modelo_rf.pkl", "preprocesador.pkl", "columnas_modelo.pkl", "variables_seleccionadas.pkl"]
    faltantes = [f for f in archivos if not os.path.exists(f)]
    if faltantes:
        return None, None, None, None
    modelo       = joblib.load("modelo_rf.pkl")
    preprocesador = joblib.load("preprocesador.pkl")
    columnas     = joblib.load("columnas_modelo.pkl")
    variables    = joblib.load("variables_seleccionadas.pkl")
    return modelo, preprocesador, columnas, variables

modelo, preprocesador, columnas_modelo, variables_seleccionadas = cargar_modelo()

# ─── DATOS AUXILIARES ─────────────────────────────────────────────────────────
PUESTOS_ES = {
    "Healthcare Representative": "Representante de Salud",
    "Human Resources": "Recursos Humanos",
    "Laboratory Technician": "Técnico de Laboratorio",
    "Manager": "Gerente",
    "Manufacturing Director": "Director de Manufactura",
    "Research Director": "Director de Investigación",
    "Research Scientist": "Científico de Investigación",
    "Sales Executive": "Ejecutivo de Ventas",
    "Sales Representative": "Representante de Ventas",
}
PUESTOS_EN = {v: k for k, v in PUESTOS_ES.items()}

SATISFACCION_LABELS = {1: "1 — Baja", 2: "2 — Media", 3: "3 — Alta", 4: "4 — Muy alta"}

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="rp-header">
  <h1>🔍 RiskPrevention</h1>
  <p>Herramienta inteligente para la predicción de riesgo de deserción laboral · Universidad Privada del Norte</p>
</div>
""", unsafe_allow_html=True)

# ─── SIDEBAR — FORMULARIO DE ENTRADA ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Datos del Trabajador")
    st.markdown("Ingresa las características del empleado a evaluar:")
    st.markdown("---")

    # Variables numéricas
    edad = st.slider("🎂 Edad", min_value=18, max_value=60, value=35, step=1)
    ingreso = st.number_input("💰 Ingreso mensual (USD)", min_value=1009, max_value=19999, value=5000, step=100)
    distancia = st.slider("🏠 Distancia al trabajo (km)", min_value=1, max_value=29, value=5, step=1)
    anios_empresa = st.slider("🏢 Años en la empresa", min_value=0, max_value=40, value=5, step=1)
    anios_exp = st.slider("💼 Años de experiencia total", min_value=0, max_value=40, value=8, step=1)
    anios_gerente = st.slider("👔 Años con el gerente actual", min_value=0, max_value=17, value=3, step=1)

    st.markdown("---")
    # Satisfacción (escalas 1-4)
    sat_laboral = st.select_slider(
        "😊 Satisfacción laboral",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: SATISFACCION_LABELS[x],
    )
    sat_ambiente = st.select_slider(
        "🌿 Satisfacción del ambiente",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: SATISFACCION_LABELS[x],
    )

    st.markdown("---")
    # Categóricas
    horas_extra_label = st.radio(
        "⏱️ ¿Realiza horas extra?",
        options=["Sí", "No"],
        index=1,
        horizontal=True,
    )
    horas_extra = "Yes" if horas_extra_label == "Sí" else "No"

    puesto_es = st.selectbox(
        "💼 Puesto de trabajo",
        options=list(PUESTOS_ES.values()),
        index=0,
    )
    puesto_en = PUESTOS_EN[puesto_es]

    st.markdown("---")
    predecir = st.button("🔎 Analizar riesgo", use_container_width=True, type="primary")

# ─── PANEL PRINCIPAL ──────────────────────────────────────────────────────────
if modelo is None:
    st.error("""
    ⚠️ **No se encontraron los archivos del modelo.**

    Para ejecutar la aplicación necesitas generar primero los artefactos del modelo ejecutando el notebook `Prediccion_Renuncia_v2.ipynb` en Google Colab.
    Luego coloca estos archivos en la misma carpeta que `app.py`:

    - `modelo_rf.pkl`
    - `preprocesador.pkl`
    - `columnas_modelo.pkl`
    - `variables_seleccionadas.pkl`
    """)
    st.stop()

# ─── PANEL IZQUIERDO: resumen de datos ingresados ─────────────────────────────
col_resumen, col_resultado = st.columns([1, 1.6], gap="large")

with col_resumen:
    st.markdown("#### 📋 Perfil del trabajador")
    datos_tabla = {
        "Característica": [
            "Edad", "Ingreso mensual", "Distancia al trabajo",
            "Años en la empresa", "Años de experiencia total",
            "Años con el gerente", "Satisfacción laboral",
            "Satisfacción del ambiente", "Horas extra", "Puesto de trabajo",
        ],
        "Valor": [
            f"{edad} años", f"USD {ingreso:,}", f"{distancia} km",
            f"{anios_empresa} años", f"{anios_exp} años",
            f"{anios_gerente} años",
            SATISFACCION_LABELS[sat_laboral],
            SATISFACCION_LABELS[sat_ambiente],
            horas_extra_label, puesto_es,
        ],
    }
    st.dataframe(
        pd.DataFrame(datos_tabla),
        use_container_width=True,
        hide_index=True,
        height=390,
    )

# ─── PANEL DERECHO: resultado de la predicción ────────────────────────────────
with col_resultado:
    if not predecir:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#9e9e9e;">
            <div style="font-size:3.5rem;">🧠</div>
            <p style="margin-top:1rem; font-size:1rem;">
                Completa el perfil del trabajador en el panel izquierdo<br>y presiona
                <strong>Analizar riesgo</strong> para obtener la predicción.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Construir el DataFrame de entrada ──────────────────────────────
        entrada = pd.DataFrame([{
            "IngresoMensual":        ingreso,
            "HorasExtra":            horas_extra,
            "SatisfaccionLaboral":   sat_laboral,
            "SatisfaccionAmbiente":  sat_ambiente,
            "Edad":                  edad,
            "AniosEnEmpresa":        anios_empresa,
            "AniosExperienciaTotal": anios_exp,
            "AniosConGerenteActual": anios_gerente,
            "PuestoTrabajo":         puesto_en,
            "DistanciaCasa":         distancia,
        }])

        # Mantener solo las columnas que el preprocesador conoce
        entrada = entrada[variables_seleccionadas]

        # Transformar
        entrada_trans = preprocesador.transform(entrada)
        entrada_df    = pd.DataFrame(entrada_trans, columns=columnas_modelo)

        # Predecir
        prediccion   = modelo.predict(entrada_df)[0]
        probabilidad = modelo.predict_proba(entrada_df)[0]
        prob_renuncia    = probabilidad[1]
        prob_permanece   = probabilidad[0]

        # ── Nivel de riesgo ────────────────────────────────────────────────
        if prob_renuncia < 0.30:
            nivel     = "BAJO"
            css_card  = "risk-bajo"
            css_text  = "bajo-text"
            emoji     = "🟢"
            bar_color = "#43a047"
        elif prob_renuncia < 0.60:
            nivel     = "MEDIO"
            css_card  = "risk-medio"
            css_text  = "medio-text"
            emoji     = "🟡"
            bar_color = "#fb8c00"
        else:
            nivel     = "ALTO"
            css_card  = "risk-alto"
            css_text  = "alto-text"
            emoji     = "🔴"
            bar_color = "#e53935"

        # ── Tarjeta de riesgo ──────────────────────────────────────────────
        st.markdown(f"""
        <div class="{css_card}">
          <p class="risk-label {css_text}">{emoji} RIESGO {nivel}</p>
          <p class="risk-sub {css_text}">
            {'El trabajador presenta señales de posible renuncia.' if prediccion == 1
             else 'El trabajador probablemente permanecerá en la empresa.'}
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Barras de probabilidad ─────────────────────────────────────────
        st.markdown("#### 📊 Probabilidades del modelo")
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de renuncia",  f"{prob_renuncia:.1%}")
        c2.metric("Probabilidad de permanencia", f"{prob_permanece:.1%}")

        st.markdown(f"""
        <div class="prob-bar-bg">
          <div class="prob-bar-fill"
               style="width:{prob_renuncia*100:.1f}%; background:{bar_color};">
          </div>
        </div>
        <p style="font-size:0.78rem; color:#888; margin-top:0.3rem;">
          Barra de riesgo de renuncia ({prob_renuncia:.1%})
        </p>
        """, unsafe_allow_html=True)

        # ── Recomendaciones ────────────────────────────────────────────────
        st.markdown("#### 💡 Recomendaciones para Recursos Humanos")
        if nivel == "BAJO":
            recomendaciones = [
                "Mantener el seguimiento periódico del bienestar del colaborador.",
                "Reconocer públicamente el buen desempeño y la permanencia.",
                "Asegurar que las condiciones laborales actuales se mantengan estables.",
                "Ofrecer oportunidades de formación o crecimiento para reforzar el vínculo.",
            ]
        elif nivel == "MEDIO":
            recomendaciones = [
                "Programar una conversación de seguimiento con el colaborador en los próximos 15 días.",
                "Revisar la carga de trabajo y el balance entre horas extra y compensación.",
                "Evaluar las oportunidades de ascenso o cambio de rol disponibles.",
                "Identificar si la relación con el equipo o el jefe directo genera fricción.",
                "Considerar un ajuste salarial si el ingreso está por debajo del mercado.",
            ]
        else:
            recomendaciones = [
                "Activar protocolo de retención: agendar entrevista urgente con el colaborador.",
                "Revisar de inmediato la política de horas extra y su retribución en la empresa.",
                "Evaluar el plan de carrera y proponer acciones concretas de ascenso o mejora salarial.",
                "Involucrar al jefe directo y al área de Recursos Humanos en un plan de acción.",
                "Considerar beneficios no salariales: flexibilidad horaria, trabajo remoto, capacitación.",
                "Documentar el riesgo en el sistema de gestión de talento para seguimiento.",
            ]

        items = "".join(f"<li>{r}</li>" for r in recomendaciones)
        st.markdown(f"""
        <div class="rec-box">
          <h4>Acciones recomendadas — Riesgo {nivel}</h4>
          <ul>{items}</ul>
        </div>
        """, unsafe_allow_html=True)

        # ── Variables más influyentes ──────────────────────────────────────
        with st.expander("📌 Ver importancia de variables del modelo"):
            importancias = pd.DataFrame({
                "Variable": columnas_modelo,
                "Importancia (%)": modelo.feature_importances_ * 100,
            }).sort_values("Importancia (%)", ascending=False).reset_index(drop=True)
            importancias["Importancia (%)"] = importancias["Importancia (%)"].round(2)
            st.dataframe(importancias, use_container_width=True, hide_index=True)

# ─── SECCIÓN INFORMATIVA ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Acerca de RiskPrevention y el modelo"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **¿Qué es RiskPrevention?**

        Es una herramienta de predicción de riesgo de deserción laboral basada en
        Machine Learning supervisado. Fue desarrollada como proyecto de curso en la
        Universidad Privada del Norte, utilizando el dataset IBM HR Analytics
        (1,470 registros, 35 características originales).

        **Algoritmo utilizado:** Random Forest (100 estimadores, `class_weight="balanced"`)

        **Dataset:** IBM HR Analytics Employee Attrition & Performance (Kaggle, 2017)
        """)
    with col_b:
        st.markdown("""
        **Variables seleccionadas (10 de 35):**

        | Variable | Tipo |
        |---|---|
        | Ingreso mensual | Numérica |
        | Horas extra | Categórica |
        | Satisfacción laboral | Ordinal |
        | Satisfacción del ambiente | Ordinal |
        | Edad | Numérica |
        | Años en la empresa | Numérica |
        | Años de experiencia total | Numérica |
        | Años con el gerente actual | Numérica |
        | Puesto de trabajo | Categórica |
        | Distancia al trabajo | Numérica |
        """)

    st.markdown("""
    **Desempeño del modelo (conjunto de prueba — 20%):**

    | Métrica | Regresión Logística | Árbol de Decisión | **Random Forest** |
    |---|---|---|---|
    | Accuracy | 70.1% | 77.9% | **82.7%** |
    | Precision | 29.7% | 28.6% | **37.5%** |
    | Recall | 63.8% | 25.5% | 12.8% |
    | F1-Score | 40.5% | 27.0% | 19.0% |
    | **ROC-AUC** | 0.733 | 0.567 | **0.765** |

    > El Random Forest fue seleccionado por su mayor ROC-AUC (0.765) y Accuracy (82.7%).
    > El desbalance de clases fue manejado mediante `class_weight="balanced"`, sin generación de datos sintéticos.
    """)

st.markdown("""
<div class="footer">
  RiskPrevention · Sistemas Inteligentes y Machine Learning · Universidad Privada del Norte · 2025<br>
  Modelo: Random Forest | Dataset: IBM HR Analytics | Framework: Streamlit
</div>
""", unsafe_allow_html=True)
