# 🛡️ Fraud Sentinel — Sistema End-to-End de Detección de Fraude en Pagos QR  

> **"Prevenir el fraude antes de que ocurra"**  

Este proyecto es una **solución industrializada de detección de fraude en transacciones QR**, desarrollada como caso de uso **end-to-end** en **Databricks**.  
Integra **ingeniería de datos, machine learning, MLOps y analítica de negocio**, siguiendo la arquitectura **Lakehouse + Medallion**.  

🔗 [App en Streamlit (demo scoring & triage)](https://deteccionfraudesqr-pw6qb4d4ptdgahbez5qmdn.streamlit.app/)  
📊 Dashboards Power BI / Tableau (*en desarrollo*)  
🧠 Modelo registrado en **Databricks MLflow + Unity Catalog**  

---

## 🚀 Problema de Negocio  

Los pagos con QR están creciendo rápidamente en América Latina y Asia, pero también lo están las tácticas de fraude:  
- **Códigos QR falsos o clonados** sobre terminales legítimas.  
- **Reutilización de QR dinámicos** en múltiples transacciones.  
- **Ataques de velocidad** (muchos pagos en pocos minutos).  
- **Pagos anómalos en horarios o ubicaciones inusuales**.  

**Objetivo**: Detectar patrones de fraude en tiempo real o en batch, **maximizando el recall** (capturar fraudes) y aplicando una **política de triage** para optimizar la gestión operativa.  

### ✅ Política de Triage
| Bandera | Acción |
|---------|--------|
| **ALTO_RIESGO** | Bloqueo automático + notificación inmediata. |
| **REVISAR** | Envío a analista → priorización según `expected_loss`. |
| **OK** | Transacción aprobada sin intervención. |

---

## 🏗️ Arquitectura del Proyecto (Lakehouse Medallion en Databricks)  

1. **Bronze** → Ingesta de datos sintéticos + dataset público Kaggle (pagos móviles).  
2. **Silver** → Limpieza, deduplicación, estandarización de currency, enmascaramiento PII.  
3. **Gold** →  
   - `fact_qr_tx_daily` (KPIs diarios).  
   - `dim_merchant`, `dim_payer`, `dim_device`.  
   - `alerts_scored` (transacciones con score + triage).  
4. **Features** → `features.qr_tx_features_v1` con variables de:  
   - Velocidad (`payer_tx_count_1h`, `payer_tx_count_24h`).  
   - Geográficas (distancia haversine).  
   - Anomalías de monto (z-score).  
   - Reutilización de QR.  
5. **ML & MLOps** →  
   - Modelos Logistic Regression y Random Forest con **GroupKFold por merchant_id**.  
   - Tracking con **MLflow** + registro en **Unity Catalog**.  
   - Thresholds optimizados para recall y expected loss.  
6. **App & BI** →  
   - **Streamlit app** para scoring por lote y triage.  
   - Dashboards en Power BI/Tableau conectados al Warehouse (`dash.*`).  

---


## 🔬 Modelos y Métricas  

- **Entrenamiento**: Logistic Regression + Random Forest (`class_weight=balanced`).  
- **Validación**: GroupKFold (agrupado por `merchant_id`), evitando fuga de datos.  
- **Métrica clave**: **PR AUC** (≥ 0.90 en validación).  
- **Inferencia batch**: job programado en Databricks que escribe en `gold.alerts_scored`.  
- **Política de triage**: thresholds versionados (`thresholds.json`).  

---

## 💻 App Streamlit (MVP Scoring)  

- Carga de CSV con columnas mínimas:  
  `amount`, `distance_km`, `payer_tx_count_1h`, `payer_tx_count_24h`, `amount_zscore_payer_7d`.  
- Salida:  
  - `fraud_score` (0–1).  
  - `expected_loss` (score × amount).  
  - `triage`: **ALTO_RIESGO / REVISAR / OK**.  
- Funciones extra:  
  - Descargar CSV con resultados.  
  - Si incluye `is_fraud`, calcula **PR AUC** y dibuja curva Precision-Recall.  

---

## 📊 Business Intelligence  

- Dashboards en Power BI / Tableau con KPIs:  
  - Total transacciones, tasa de fraude, monto fraudulento.  
  - Serie temporal (día/semana/mes).  
  - Mapas por merchant con fraude geolocalizado.  
  - Top merchants/payers por expected_loss.  
  - Histogramas/heatmaps por `qr_type`, `mcc`, canal, hora.  

---

## 🔒 Seguridad & Gobernanza  

- **Unity Catalog**: permisos por rol, column masking, row filters.  
- **Costos**: partición por fecha, OPTIMIZE ZORDER, auto-stop clusters.  
- **Observabilidad**: freshness de datos, drift de features, alertas de scoring.  

---

## ✅ Estado del Proyecto  

- [x] Generación de datasets sintéticos.  
- [x] Construcción de Lakehouse (Bronze/Silver/Gold).  
- [x] Feature Store v1.  
- [x] Entrenamiento + registro de modelos en UC.  
- [x] App Streamlit MVP.  
- [ ] Dashboards BI (Power BI / Tableau).  
- [ ] Documentación final (data contracts, runbook).  

---

## 🤝 Contribuciones  

Este proyecto es parte de mi portfolio en **fraude y riesgo financiero**.  
Comentarios y sugerencias son bienvenidos vía issues o PRs.  

## 👤 Autor

**Pedro Rubio** — Machine Learning Engenieer 

- App Streamlit *https://deteccionfraudesqr-rif5f8gnfhmduf6rm9eyqp.streamlit.app/*, BI con Tableau/Power BI.
- Drive/Model, Artifacts: *https://colab.research.google.com/drive/1_Ed65bITdC714VqEDTFk9ouYxYGiwDoL?usp=sharing*
- Databricks/Notebook & Artefactos *https://dbc-d087d100-620e.cloud.databricks.com/browse/folders/2447160281203034?o=2884050240173164*
- Contacto: *srdelosdatos@gmail.com* — *www.linkedin.com/in/srdelosdatos* — 
