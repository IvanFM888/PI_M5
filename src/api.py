# src/api.py

# 1. IMPORTACIONES TÉCNICAS
import sys  # Para manipular la configuración interna de Python
import os   # Para navegar por carpetas del sistema operativo

# --- CORRECCIÓN DE RUTAS (Truco importante) ---
# A veces, cuando ejecutamos la API, Python "no ve" el archivo ft_engineering.py porque está buscando en la carpeta equivocada. Esta línea le dice busca también en la carpeta actual
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 2. IMPORTACIONES DE LA API Y DATOS
from fastapi import FastAPI, HTTPException # FastAPI es el constructor del servidor web
import pandas as pd     # Para convertir los datos recibidos en una tabla
import joblib           # Para "descongelar" el modelo guardado
from pydantic import BaseModel # Para validar que los datos lleguen correctamente (el "portero")

# Importamos nuestro módulo de ingeniería.
# El bloque try/except es un "paracaídas": intenta importarlo normal, 
# y si falla, intenta buscarlo dentro de la carpeta 'src'.
# ¿Por qué lo necesitamos? Porque el preprocesador guardado depende de funciones que están ahí.
try:
    import ft_engineering as ft
except ImportError:
    from src import ft_engineering as ft

# 3. INICIALIZACIÓN DE LA APP
# Aquí "prendemos las luces" del servidor.
app = FastAPI(
    title="API de Riesgo Crediticio 🏦",
    description="Endpoint para predecir si un cliente pagará a tiempo.",
    version="1.0.0"
)

# --- 4. CARGA DEL CEREBRO (Artefactos) ---
# Esto se ejecuta UNA sola vez cuando inicias el servidor (el arranque).
# Busca los archivos .pkl que guardó el script 'save_model.py'.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
PREP_PATH = os.path.join(BASE_DIR, 'preprocessor.pkl')

print("Cargando modelo y preprocesador...")
try:
    # Descongelamos la inteligencia artificial
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREP_PATH)
    print("Sistema listo para predecir.")
except Exception as e:
    # Si no encuentra los archivos, imprime el error pero no rompe la app (aunque no podrá predecir).
    print(f"Error crítico cargando modelos: {e}")
    model = None
    preprocessor = None

# --- 5. DEFINICIÓN DEL FORMULARIO (Schema) ---
# Esta clase actúa como un "filtro" de seguridad.
# Si alguien intenta enviar texto en el campo 'salario', la API le dará error automáticamente.
# Pydantic asegura que los datos sean del tipo correcto antes de pasar al modelo.
class ClientData(BaseModel):
    # Variables Numéricas (deben ser números enteros o decimales)
    capital_prestado: float
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    promedio_ingresos_datacredito: float
    plazo_meses: int
    edad_cliente: int
    cant_creditosvigentes: int
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    
    # Variables de Texto (Strings)
    tipo_laboral: str
    tendencia_ingresos: str

# --- 6. ENDPOINT DE PREDICCIÓN (La Ventanilla de Servicio) ---
# Cuando alguien envíe datos a la dirección "/predict", se ejecuta esta función.
# @app.post significa que estamos enviando información al servidor.
@app.post("/predict")
def predict_credit_risk(client: ClientData):
    
    # Verificación de seguridad: ¿El modelo cargó bien al inicio?
    if not model or not preprocessor:
        raise HTTPException(status_code=500, detail="El modelo no está cargado en el servidor.")
    
    try:
        # PASO A: Convertir el JSON recibido a una tabla de Pandas (DataFrame)
        # client.dict() convierte el objeto que recibimos a un diccionario normal.
        input_data = pd.DataFrame([client.dict()])
        
        # PASO B: Preprocesamiento (Limpieza)
        # Convertimos explícitamente a string las categóricas para evitar confusiones.
        for col in ['tipo_laboral', 'tendencia_ingresos']:
            input_data[col] = input_data[col].astype(str)
            
        # Usamos el preprocesador que cargamos para transformar los datos
        # (Aplica las mismas reglas, promedios y escalas que usamos al entrenar).
        # ¡OJO! Usamos .transform(), NUNCA .fit() aquí (no queremos aprender de un solo cliente).
        X_processed = preprocessor.transform(input_data)
        
        # PASO C: Predicción (Consultar al Modelo)
        # .predict devuelve [0] o [1]. Tomamos el primer valor con [0].
        prediction = model.predict(X_processed)[0] 
        # .predict_proba devuelve la probabilidad [Prob_No, Prob_Si]. Tomamos la del 'Si' con [1].
        probability = model.predict_proba(X_processed)[0][1] 
        
        # PASO D: Construir la respuesta
        # Traducimos el 1/0 a algo que entienda un humano.
        result = "Aprobado" if prediction == 1 else "Rechazado"
        
        # Devolvemos un diccionario que FastAPI convertirá automáticamente a JSON para el usuario.
        return {
            "prediction": result,
            "probability_pago_atiempo": float(round(probability, 4)),
            "risk_level": "Bajo" if probability > 0.7 else "Alto" # Lógica de negocio extra
        }
        
    except Exception as e:
        # Si algo falla (ej: datos corruptos), devolvemos un error 400 (Bad Request).
        raise HTTPException(status_code=400, detail=f"Error procesando datos: {str(e)}")

# --- 7. ENDPOINT DE PRUEBA (Health Check) ---
# Una ruta simple para ver si el servidor está vivo sin enviar datos complejos.
# Se accede entrando a la raíz "/" de la web.
@app.get("/")
def home():
    return {"message": "API de Riesgo Crediticio funcionando correctamente 🚀"}