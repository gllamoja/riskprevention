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
    .rp-header {
        background: linear-gradient(135deg, #1a237e 0%, #283593 50%, #1565c0 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .rp-header h1 { margin: 0; font-size: 2rem; font-weight: 700; }
    .rp-header p  { margin: 0.4rem 0 0; font-size: 1rem; opacity: 0.85; }

    .risk-bajo  { background:#e8f5e9; border:2px solid #43a047; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-medio { background:#fff8e1; border:2px solid #fb8c00; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-alto  { background:#ffebee; border:2px solid #e53935; border-radius:12px; padding:1.5rem; text-align:center; }
    .risk-label { font-size:2rem; font-weight:800; margin:0; }
    .risk-sub   { font-size:0.95rem; margin:0.3rem 0 0; }
    .bajo-text  { color:#2e7d32; }
    .medio-text { color:#e65100; }
    .alto-text  { color:#b71c1c; }

    .prob-bar-bg   { background:#e0e0e0; border-radius:8px; height:20px; margin:0.5rem 0; }
    .prob-bar-fill { height:20px; border-radius:8px; }

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

    section[data-testid="stSidebar"] { background: #f0f4ff; }
    .footer { text-align:center; color:#999; font-size:0.78rem; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)

# ─── COLUMNAS DEL MODELO (el orden exacto con que fue entrenado) ──────────────
# OneHotEncoder(drop="first") sobre HorasExtra y PuestoTrabajo, luego passthrough
# HorasExtra: ["No","Yes"] → drop "No" → columna "HorasExtra_Yes"
# PuestoTrabajo: 9 roles ordenados → drop "Healthcare Representative" → 8 columnas
COLUMNAS_MODELO = [
    "categoricas__HorasExtra_Yes",
    "categoricas__PuestoTrabajo_Human Resources",
    "categoricas__PuestoTrabajo_Laboratory Technician",
    "categoricas__PuestoTrabajo_Manager",
    "categoricas__PuestoTrabajo_Manufacturing Director",
    "categoricas__PuestoTrabajo_Research Director",
    "categoricas__PuestoTrabajo_Research Scientist",
    "categoricas__PuestoTrabajo_Sales Executive",
    "categoricas__PuestoTrabajo_Sales Representative",
    "remainder__IngresoMensual",
    "remainder__SatisfaccionLaboral",
    "remainder__SatisfaccionAmbiente",
    "remainder__Edad",
    "remainder__AñosEnEmpresa",
    "remainder__AñosExperienciaTotal",
    "remainder__AñosConGerenteActual",
    "remainder__DistanciaCasa",
]

PUESTOS_DUMMIES = [
    "Human Resources",
    "Laboratory Technician",
    "Manager",
    "Manufacturing Director",
    "Research Director",
    "Research Scientist",
    "Sales Executive",
    "Sales Representative",
]

# ─── CARGA DEL MODELO ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando modelo de predicción...")
def cargar_modelo():
    if not os.path.exists("modelo_rf.pkl"):
        return None
    return joblib.load("modelo_rf.pkl")

modelo = cargar_modelo()

# ─── FUNCIÓN DE ENCODING MANUAL ───────────────────────────────────────────────
def preparar_entrada(edad, ingreso, distancia, anios_empresa, anios_exp,
                     anios_gerente, sat_laboral, sat_ambiente,
                     horas_extra_en, puesto_en):
    fila = {}
    # HorasExtra (1 columna)
    fila["categoricas__HorasExtra_Yes"] = 1 if horas_extra_en == "Yes" else 0
    # PuestoTrabajo (8 columnas, drop="first" eliminó "Healthcare Representative")
    for p in PUESTOS_DUMMIES:
        fila[f"categoricas__PuestoTrabajo_{p}"] = 1 if puesto_en == p else 0
    # Numéricas (passthrough, en el orden que tenían en X_train)
    fila["remainder__IngresoMensual"]        = ingreso
    fila["remainder__SatisfaccionLaboral"]   = sat_laboral
    fila["remainder__SatisfaccionAmbiente"]  = sat_ambiente
    fila["remainder__Edad"]                  = edad
    fila["remainder__AñosEnEmpresa"]         = anios_empresa
    fila["remainder__AñosExperienciaTotal"]  = anios_exp
    fila["remainder__AñosConGerenteActual"]  = anios_gerente
    fila["remainder__DistanciaCasa"]         = distancia
    return pd.DataFrame([fila])[COLUMNAS_MODELO]

# ─── DATOS AUXILIARES ─────────────────────────────────────────────────────────
PUESTOS_ES = {
    "Healthcare Representative": "Representante de Salud",
    "Human Resources":           "Recursos Humanos",
    "Laboratory Technician":     "Técnico de Laboratorio",
    "Manager":                   "Gerente",
    "Manufacturing Director":    "Director de Manufactura",
    "Research Director":         "Director de Investigación",
    "Research Scientist":        "Científico de Investigación",
    "Sales Executive":           "Ejecutivo de Ventas",
    "Sales Representative":      "Representante de Ventas",
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

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Datos del Trabajador")
    st.markdown("Ingresa las características del empleado a evaluar:")
    st.markdown("---")

    edad          = st.slider("🎂 Edad", 18, 60, 35)
    ingreso       = st.number_input("💰 Ingreso mensual (USD)", 1009, 19999, 5000, 100)
    distancia     = st.slider("🏠 Distancia al trabajo (km)", 1, 29, 5)
    anios_empresa = st.slider("🏢 Años en la empresa", 0, 40, 5)
    anios_exp     = st.slider("💼 Años de experiencia total", 0, 40, 8)
    anios_gerente = st.slider("👔 Años con el gerente actual", 0, 17, 3)

    st.markdown("---")
    sat_laboral  = st.select_slider("😊 Satisfacción laboral",
                                    options=[1,2,3,4], value=3,
                                    format_func=lambda x: SATISFACCION_LABELS[x])
    sat_ambiente = st.select_slider("🌿 Satisfacción del ambiente",
                                    options=[1,2,3,4], value=3,
                                    format_func=lambda x: SATISFACCION_LABELS[x])

    st.markdown("---")
    horas_extra_label = st.radio("⏱️ ¿Realiza horas extra?",
                                 ["Sí", "No"], index=1, horizontal=True)
    horas_extra_en = "Yes" if horas_extra_label == "Sí" else "No"

    puesto_es = st.selectbox("💼 Puesto de trabajo",
                             list(PUESTOS_ES.values()), index=0)
    puesto_en = PUESTOS_EN[puesto_es]

    st.markdown("---")
    predecir = st.button("🔎 Analizar riesgo", use_container_width=True, type="primary")

# ─── PANEL PRINCIPAL ──────────────────────────────────────────────────────────
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
            SATISFACCION_LABELS[sat_laboral], SATISFACCION_LABELS[sat_ambiente],
            horas_extra_label, puesto_es,
        ],
    }
    st.dataframe(pd.DataFrame(datos_tabla), use_container_width=True,
                 hide_index=True, height=390)

with col_resultado:
    if modelo is None:
        st.error("⚠️ No se encontró `modelo_rf.pkl`. "
                 "Ejecuta el notebook en Colab y sube el archivo a GitHub.")
    elif not predecir:
        st.markdown("""
        <div style="text-align:center; padding:3rem 1rem; color:#9e9e9e;">
            <div style="font-size:3.5rem;">🧠</div>
            <p style="margin-top:1rem; font-size:1rem;">
                Completa el perfil en el panel izquierdo<br>
                y presiona <strong>Analizar riesgo</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        entrada = preparar_entrada(
            edad, ingreso, distancia, anios_empresa, anios_exp,
            anios_gerente, sat_laboral, sat_ambiente, horas_extra_en, puesto_en
        )

        prediccion   = modelo.predict(entrada)[0]
        probabilidad = modelo.predict_proba(entrada)[0]
        prob_renuncia  = probabilidad[1]
        prob_permanece = probabilidad[0]

        if prob_renuncia < 0.30:
            nivel, css_card, css_text, emoji, bar_color = \
                "BAJO", "risk-bajo", "bajo-text", "🟢", "#43a047"
        elif prob_renuncia < 0.60:
            nivel, css_card, css_text, emoji, bar_color = \
                "MEDIO", "risk-medio", "medio-text", "🟡", "#fb8c00"
        else:
            nivel, css_card, css_text, emoji, bar_color = \
                "ALTO", "risk-alto", "alto-text", "🔴", "#e53935"

        st.markdown(f"""
        <div class="{css_card}">
          <p class="risk-label {css_text}">{emoji} RIESGO {nivel}</p>
          <p class="risk-sub {css_text}">
            {'El trabajador presenta señales de posible renuncia.'
             if prediccion == 1 else
             'El trabajador probablemente permanecerá en la empresa.'}
          </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### 📊 Probabilidades del modelo")
        c1, c2 = st.columns(2)
        c1.metric("Probabilidad de renuncia",    f"{prob_renuncia:.1%}")
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

        st.markdown("#### 💡 Recomendaciones para Recursos Humanos")
        if nivel == "BAJO":
            recs = [
                "Mantener el seguimiento periódico del bienestar del colaborador.",
                "Reconocer públicamente el buen desempeño y la permanencia.",
                "Ofrecer oportunidades de formación para reforzar el vínculo.",
            ]
        elif nivel == "MEDIO":
            recs = [
                "Programar una conversación de seguimiento en los próximos 15 días.",
                "Revisar la carga de trabajo y la compensación por horas extra.",
                "Evaluar oportunidades de ascenso o cambio de rol.",
                "Identificar si la relación con el equipo o jefe genera fricción.",
            ]
        else:
            recs = [
                "Activar protocolo de retención: agendar entrevista urgente.",
                "Revisar de inmediato la política de horas extra y su retribución.",
                "Proponer acciones concretas de mejora salarial o ascenso.",
                "Involucrar al jefe directo y a Recursos Humanos en un plan de acción.",
                "Considerar beneficios no salariales: flexibilidad, trabajo remoto.",
            ]

        items = "".join(f"<li>{r}</li>" for r in recs)
        st.markdown(f"""
        <div class="rec-box">
          <h4>Acciones recomendadas — Riesgo {nivel}</h4>
          <ul>{items}</ul>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("📌 Ver importancia de variables del modelo"):
            imp = pd.DataFrame({
                "Variable":       COLUMNAS_MODELO,
                "Importancia (%)": (modelo.feature_importances_ * 100).round(2),
            }).sort_values("Importancia (%)", ascending=False).reset_index(drop=True)
            st.dataframe(imp, use_container_width=True, hide_index=True)

# ─── SECCIÓN INFORMATIVA ──────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ Acerca de RiskPrevention y el modelo"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **¿Qué es RiskPrevention?**

        Herramienta de predicción de riesgo de deserción laboral basada en
        Machine Learning supervisado. Desarrollada como proyecto de curso en la
        Universidad Privada del Norte, con el dataset IBM HR Analytics
        (1,470 registros, 35 características originales).

        **Algoritmo:** Random Forest (100 estimadores, `class_weight="balanced"`)
        """)
    with col_b:
        st.markdown("""
        **Desempeño del modelo (conjunto de prueba — 20%):**

        | Métrica | Valor |
        |---|---|
        | Accuracy | 82.7% |
        | Recall | 12.8% |
        | F1-Score | 19.0% |
        | **ROC-AUC** | **0.765** |
        """)

st.markdown("""
<div class="footer">
  RiskPrevention · Sistemas Inteligentes y Machine Learning · Universidad Privada del Norte · 2025
</div>
""", unsafe_allow_html=True)
