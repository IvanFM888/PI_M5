## **Proyecto Integrador M5 - Avance 1.**
### **Comprensión y EDA**

Alumno: Ivan Martinez

Cohorte: DSFT01

---

# 🏦 Detección de Riesgo Crediticio y Monitoreo de Data Drift

## 📋 Descripción del Caso de Negocio
Este proyecto implementa una solución MLOps end-to-end para una entidad financiera. El objetivo es predecir la probabilidad de que un cliente pague su crédito a tiempo (**Pago_atiempo**), basándose en su perfil demográfico y financiero.

Además del modelado, se ha implementado un sistema robusto de **monitoreo** para detectar cambios en el comportamiento de los clientes (Data Drift) que puedan degradar la calidad de las predicciones en producción.

## 🚀 Estructura del Proyecto
El proyecto sigue una arquitectura modular:
- `src/ft_engineering.py`: Pipeline de transformación de datos (imputación, logaritmos, encoding).
- `src/model_training.py`: Entrenamiento y evaluación de modelos (Random Forest, Gradient Boosting).
- `src/app_monitoring.py`: Dashboard interactivo en Streamlit para detección de Drift.

## 📊 Hallazgos Principales (Avance 2)
1. **Modelado:** Se compararon tres algoritmos. El modelo **Gradient Boosting** mostró el mejor balance entre AUC y estabilidad.
2. **Corrección de Data Leakage:** Inicialmente se detectó un AUC de 1.00 debido a variables que revelaban el futuro (mora). Se eliminaron variables como `saldo_mora` y `puntaje` para obtener un modelo predictivo real (AUC ~0.64).

## 🕵️‍♂️ Sistema de Monitoreo (Avance 3)
Se desarrolló un Dashboard en **Streamlit** que evalúa periódicamente la salud de los datos utilizando métricas estadísticas:
- **PSI (Population Stability Index):** Alerta temprana de cambios en la distribución.
- **Test KS (Kolmogorov-Smirnov):** Detección de cambios en la forma de los datos numéricos.
- **Chi-Cuadrado:** Validación de cambios en frecuencias de categorías.
- **JSD (Jensen-Shannon):** Medición de divergencia entre distribuciones.

### Semáforo de Riesgo
- 🟢 **Estable:** PSI < 0.1
- 🟡 **Alerta:** 0.1 <= PSI < 0.25
- 🔴 **Crítico:** PSI >= 0.25 (Requiere re-entrenamiento)

## 💻 Cómo ejecutar
1. **Entrenamiento:** `python src/model_training.py`
2. **Monitoreo:** `streamlit run src/app_monitoring.py`

