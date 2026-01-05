## **Proyecto Integrador M5 - Avance 1.**
### **Comprensión y EDA**

Alumno: Ivan Martinez

Cohorte: DSFT01

---

# Detección de Riesgo Crediticio y Monitoreo de Data Drift

## Descripción del Caso de Negocio
Este proyecto implementa una solución MLOps end-to-end para una entidad financiera. El objetivo es predecir la probabilidad de que un cliente pague su crédito a tiempo (**Pago_atiempo**), basándose en su perfil demográfico y financiero.

Además del modelado, se ha implementado un sistema robusto de **monitoreo** para detectar cambios en el comportamiento de los clientes (Data Drift) que puedan degradar la calidad de las predicciones en producción.

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```text
PROYECTO/
├── Base_de_datos.xlsx       # Archivo con los datos históricos de clientes
├── requirements.txt         # Lista de librerías necesarias
├── README.md                # Este archivo de documentación
└── src/                     # Código Fuente
    ├── ft_engineering.py    # Modulo de limpieza y transformación de datos
    ├── model_training.py    # Script para comparar modelos y elegir el mejor
    ├── save_model.py        # Script para entrenar y guardar el modelo final (.pkl)
    ├── model_monitoring.py  # Logica matemática para detectar Drift (cambios en datos)
    ├── app_monitoring.py    # Dashboard visual para monitoreo (Streamlit)
    ├── api.py               # Servidor web para predicciones (FastAPI)
    ├── best_model.pkl       # (Generado automático) El modelo entrenado
    └── preprocessor.pkl     # (Generado automático) El transformador de datos


Asistente de programación
Claro, entiendo perfectamente. Aquí tienes la versión del README.md más formal, limpia, sin emojis y centrada únicamente en los comandos necesarios para las librerías.

Markdown
# Sistema de Prediccion de Riesgo Crediticio

Este proyecto implementa un flujo completo de Machine Learning (ML) para predecir si un cliente pagará su crédito a tiempo o no.

El sistema incluye desde la limpieza de datos hasta la puesta en producción con una API y un tablero de monitoreo para vigilar la estabilidad del modelo.

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```text
PROYECTO/
├── Base_de_datos.xlsx       # Archivo con los datos históricos de clientes
├── requirements.txt         # Lista de librerías necesarias
├── README.md                # Este archivo de documentación
└── src/                     # Código Fuente
    ├── ft_engineering.py    # Modulo de limpieza y transformación de datos
    ├── model_training.py    # Script para comparar modelos y elegir el mejor
    ├── save_model.py        # Script para entrenar y guardar el modelo final (.pkl)
    ├── model_monitoring.py  # Logica matemática para detectar Drift (cambios en datos)
    ├── app_monitoring.py    # Dashboard visual para monitoreo (Streamlit)
    ├── api.py               # Servidor web para predicciones (FastAPI)
    ├── best_model.pkl       # (Generado automático) El modelo entrenado
    └── preprocessor.pkl     # (Generado automático) El transformador de datos

## Instalacion de Librerias
Instalar las dependencias necesarias:

pip install pandas numpy scikit-learn matplotlib seaborn openpyxl scipy streamlit plotly fastapi uvicorn joblib pydantic

## Guia de Uso
Sigue este orden para ejecutar el flujo del proyecto correctamente:

Paso 1: Entrenamiento y Seleccion (model_training.py)
Ejecuta este script para ver una comparativa entre varios modelos (Regresión Logística, Random Forest, etc.) y visualizar sus métricas de rendimiento.

python src/model_training.py

Paso 2: Guardar el Modelo (save_model.py)
Una vez decidido el mejor modelo, este script lo entrena con todos los datos disponibles y genera los archivos .pkl necesarios para la API.


python src/save_model.py

Nota: Verás dos nuevos archivos en la carpeta src/: best_model.pkl y preprocessor.pkl.

Paso 3: Iniciar la API (api.py)
Levanta el servidor local para empezar a recibir predicciones.

uvicorn src.api:app --reload
La API estará disponible en: https://www.google.com/search?q=http://127.0.0.1:8000

Documentación automática (Swagger): https://www.google.com/search?q=http://127.0.0.1:8000/docs

Paso 4: Monitoreo (app_monitoring.py)
Para visualizar el tablero de control y detectar si los datos han sufrido cambios significativos (Data Drift), ejecuta:

streamlit run src/app_monitoring.py

Descripcion de los Modulos
1. Ingenieria de Caracteristicas (ft_engineering.py)
Es el módulo base. Se encarga de cargar el Excel, limpiar valores nulos, convertir texto a números y prevenir la fuga de datos (Data Leakage). Este archivo es importado por los demás scripts.

2. Entrenamiento (model_training.py)
Evalúa múltiples algoritmos usando Validación Cruzada. Genera gráficos comparativos para entender qué modelo tiene mejor balance entre precisión (AUC) y estabilidad.

3. API (api.py)
Utiliza FastAPI. Carga el modelo guardado y expone un endpoint /predict. Incluye validación de datos con Pydantic para asegurar que la información recibida tenga el formato correcto.

Ejemplo de cuerpo para la petición (JSON):

JSON
{
  "capital_prestado": 5000000,
  "salario_cliente": 2500000,
  "total_otros_prestamos": 100000,
  "cuota_pactada": 150000,
  "promedio_ingresos_datacredito": 2400000,
  "plazo_meses": 36,
  "edad_cliente": 30,
  "cant_creditosvigentes": 2,
  "creditos_sectorFinanciero": 1,
  "creditos_sectorCooperativo": 0,
  "creditos_sectorReal": 1,
  "tipo_laboral": "Independiente",
  "tendencia_ingresos": "Estable"
}

4. Monitoreo (model_monitoring.py y app_monitoring.py)
Utiliza métricas estadísticas avanzadas para asegurar la calidad del modelo en el tiempo:

PSI (Population Stability Index): Para detectar cambios en variables numéricas.

Chi-Cuadrado: Para detectar cambios en variables categóricas.

Streamlit: Interfaz gráfica para visualizar las alertas de estabilidad.