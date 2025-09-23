# app.py
import streamlit as st
import pandas as pd
import joblib
import json
import os
from io import StringIO
import requests # ¡Importante añadir esta librería!

# --- Configuración de la Página ---
st.set_page_config(
    page_title="🛡️ Fraud Sentinel — QR Payment Scoring",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Personalizados ---
st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# --- URLs de los Artefactos ---
# Se recomienda guardar estas URLs como secretos en Streamlit Cloud
MODEL_URL = "https://drive.google.com/uc?export=download&id=11BhOTvi6K6Mwgorzxb6y57AIWi4kW7fF"
THRESHOLDS_URL = "https://drive.google.com/uc?export=download&id=1NoBcABkU0Dyf18O1njartMmJNUiHCVyV" 

MODEL_LOCAL_PATH = "modelo.pkl"
THRESHOLDS_LOCAL_PATH = "thresholds_RandomForest.json"

# --- Título ---
st.title("🛡️ Fraud Sentinel — QR Payment Scoring")
st.markdown("Carga un archivo CSV con transacciones QR y obtén un score de fraude + política de triage.")

# --- Sidebar: Simulación de Costos ---
st.sidebar.header("⚙️ Configuración de Triage")
c_fp = st.sidebar.slider("Costo de Falso Positivo (investigación)", 1, 100, 10)
c_fn = st.sidebar.slider("Costo de Falso Negativo (fraude perdido)", 100, 10000, 500)

# --- Función para descargar archivos ---
def download_file(url, local_filename, file_description):
    """Descarga un archivo desde una URL si no existe localmente."""
    if not os.path.exists(local_filename):
        st.info(f"📥 Descargando {file_description}, por favor espera...")
        try:
            r = requests.get(url, allow_redirects=True, timeout=60) # Añadido timeout
            r.raise_for_status()
            with open(local_filename, "wb") as f:
                f.write(r.content)
            st.success(f"✅ ¡{file_description} descargado con éxito!")
        except requests.exceptions.RequestException as e:
            st.error(f"❌ Error al descargar {file_description}: {e}")
            st.stop()

# --- Cargar Modelo y Thresholds ---
@st.cache_resource
def load_model_and_thresholds():
    """
    Descarga el modelo y los umbrales desde sus URLs si no existen localmente,
    luego los carga en memoria.
    """
    download_file(MODEL_URL, MODEL_LOCAL_PATH, "el modelo")
    download_file(THRESHOLDS_URL, THRESHOLDS_LOCAL_PATH, "los umbrales")
    
    try:
        pipeline = joblib.load(MODEL_LOCAL_PATH)
        with open(THRESHOLDS_LOCAL_PATH, "r") as f:
            thresholds = json.load(f)
        
        optimal_threshold = thresholds.get("optimal_threshold", 0.5)
        return pipeline, optimal_threshold
    except Exception as e:
        st.error(f"❌ Error al cargar los artefactos locales: {str(e)}")
        return None, 0.5

pipeline, optimal_threshold = load_model_and_thresholds()

if pipeline is None:
    st.stop()

# --- Cargar Archivo CSV ---
uploaded_file = st.file_uploader("📂 Cargar archivo CSV con transacciones", type=["csv"])

if uploaded_file is not None:
    # Leer CSV
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    df = pd.read_csv(stringio)
    
    st.success(f"✅ Archivo cargado: {uploaded_file.name} ({len(df):,} registros)")
    
    # Verificar columnas requeridas
    required_cols = [
        "amount", "distance_km", "payer_tx_count_1h",
        "payer_tx_count_24h", "amount_zscore_payer_7d"
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Faltan columnas requeridas: {missing_cols}")
        st.stop()
    
    # Predecir
    try:
        X = df[required_cols]
        scores = pipeline.predict_proba(X)[:, 1]
        df['fraud_score'] = scores
        df['expected_loss'] = df['fraud_score'] * df['amount']
        
        # Política de Triage
        df['triage'] = "OK"
        df.loc[df['fraud_score'] >= optimal_threshold, 'triage'] = "ALTO_RIESGO"
        df.loc[(df['fraud_score'] >= 0.3) & (df['fraud_score'] < optimal_threshold), 'triage'] = "REVISAR"
        
        # Ordenar por expected_loss (para REVISAR)
        df_revisar = df[df['triage'] == "REVISAR"].sort_values('expected_loss', ascending=False)
        top_n = min(50, len(df_revisar))
        df_revisar_top = df_revisar.head(top_n)
        df.loc[df_revisar_top.index, 'triage'] = "REVISAR (Top Impacto)"
        
        st.success("✅ ¡Scoring completado!")
        
        # Mostrar resumen
        st.subheader("📊 Resumen de Resultados")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transacciones", len(df))
        with col2:
            st.metric("ALTO_RIESGO", len(df[df['triage'] == "ALTO_RIESGO"]))
        with col3:
            st.metric("REVISAR", len(df[df['triage'].str.contains("REVISAR")]))
        
        # Mostrar tabla
        st.subheader("📋 Resultados Detallados")
        st.dataframe(
            df[[
                "fraud_score", "triage", "expected_loss",
                "amount", "distance_km", "payer_tx_count_1h"
            ]].sort_values("fraud_score", ascending=False),
            use_container_width=True
        )
        
        # Descargar resultados
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resultados (CSV)",
            data=csv,
            file_name='fraud_scoring_results.csv',
            mime='text/csv',
        )
        
        # Si tiene etiquetas reales, mostrar métricas
        if 'is_fraud' in df.columns:
            from sklearn.metrics import average_precision_score, precision_recall_curve
            import matplotlib.pyplot as plt
            
            st.subheader("📈 Métricas de Desempeño (si se proporcionó is_fraud)")
            pr_auc = average_precision_score(df['is_fraud'], df['fraud_score'])
            st.metric("PR AUC", f"{pr_auc:.4f}")
            
            # Curva PR
            precision, recall, thresholds_pr = precision_recall_curve(df['is_fraud'], df['fraud_score'])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(recall, precision, label=f'PR Curve (AUC = {pr_auc:.4f})')
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title('Precision-Recall Curve')
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
    except Exception as e:
        st.error(f"❌ Error durante el scoring: {str(e)}")
else:
    st.info("👆 Por favor, carga un archivo CSV para comenzar.")
    st.markdown("""
    ### 📝 Formato Esperado del CSV
    El archivo debe contener al menos estas columnas:
    - `amount`
    - `distance_km`
    - `payer_tx_count_1h`
    - `payer_tx_count_24h`
    - `amount_zscore_payer_7d`
    
    Opcional: `is_fraud` (para calcular métricas de desempeño).
    """)
