# src/save_model.py

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import joblib   # Joblib: Permite tomar la "memoria" del modelo y congelarla en un archivo para que pueda usarse después.
import os       # Para que las rutas de los archivos funcionen en cualquier PC.
from sklearn.ensemble import GradientBoostingClassifier

# Importamos el script de ingeniería que creamos antes.
import ft_engineering as ft

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def train_and_save():
    """
    Esta función hace el trabajo final:
    1. Entrena el modelo definitivo.
    2. Guarda dos archivos .pkl (el modelo y el preprocesador).
    """
    
    # 1. CARGAR DATOS
    # Usamos os.path para encontrar la carpeta base automáticamente.
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_archivo = os.path.join(ruta_base, "Base_de_datos.xlsx")
    
    # Llamamos a la función de carga segura
    df = ft.load_data(ruta_archivo)
    target = 'Pago_atiempo' # Lo que queremos predecir
    
    # 2. INGENIERÍA DE CARACTERÍSTICAS
    print("Procesando datos...")    
    # ft.perform_engineering nos devuelve los datos (X, y) PERO TAMBIÉN el 'preprocessor'.
    # El 'preprocessor' contiene las reglas matemáticas (ej: el promedio de edad es 35).
    # Necesitamos esas reglas para aplicarlas a los futuros clientes.
    X_train, X_test, y_train, y_test, preprocessor = ft.perform_engineering(df, target)
    
    # 3. ENTRENAMIENTO FINAL
    # En el paso anterior (training) decidimos que GradientBoosting era el mejor.
    # Ahora lo configuramos directamente aquí.
    print("Entrenando modelo (GradientBoosting)...")
    
    # Configuramos el "cerebro" del modelo
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # Le enseñamos con los datos procesados (.fit)
    model.fit(X_train, y_train)
    
    # 4. GUARDADO DE ARTEFACTOS (.pkl)
    # Aquí convertimos el código en memoria en archivos físicos.
    print("Guardando archivos .pkl...")
    
    # Guardamos el MODELO (quien toma la decisión de aprobar/rechazar)
    joblib.dump(model, os.path.join(ruta_base, 'src', 'best_model.pkl'))
    
    # Guardamos el PREPROCESADOR (quien limpia los datos crudos)
    # Sin este archivo, el modelo no entendería los datos nuevos que le lleguen.
    joblib.dump(preprocessor, os.path.join(ruta_base, 'src', 'preprocessor.pkl'))
    
    print("Archivos creados exitosamente en la carpeta /src")

# Bloque de ejecución: Esto hace que la función arranque al ejecutar el archivo.
if __name__ == "__main__":
    train_and_save()