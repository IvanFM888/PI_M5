# src/experiment_tracking.py
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import cross_validate
import warnings
import os

# Importamos los módulos de las fases anteriores
import ft_engineering as ft
import model_training as mt 

# Ignoramos advertencias de versión para mantener la consola limpia
warnings.filterwarnings("Ignore")

def run_experiment(experiment_name, df, target_col):
    """
    Entrena modelos y registra todo (parámetros, métricas y artefactos) en MLflow.
    """
    # 1. Configurar el Experimento en MLflow
    mlflow.set_experiment(experiment_name)
    print(f"\n Iniciando experimento: {experiment_name}")
    
    # 2. Preparar datos (Usando el pipeline de ingeniería)
    print(" Procesando datos...")
    X_train, X_test, y_train, y_test, preprocessor = ft.perform_engineering(df, target_col)
    
    # 3. Obtener la lista de modelos definidos en model_training.py
    models = mt.build_models()
    
    for name, model in models:
        # Iniciamos un "Run" (una ejecución específica)
        with mlflow.start_run(run_name=f"{name}_V1"):
            print(f"   --> Registrando: {name}")
            
            # REGISTRO DE PARÁMETROS
            # Guardamos qué configuración usó el modelo
            mlflow.log_param("model_type", name)
            
            # Registramos hiperparámetros si existen
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_iter"):
                mlflow.log_param("max_iter", model.max_iter)

            # REGISTRO DE MÉTRICAS (Validación Cruzada)
            cv_results = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy')
            mlflow.log_metric("cv_accuracy_mean", cv_results['test_score'].mean())
            mlflow.log_metric("cv_accuracy_std", cv_results['test_score'].std())
            
            # ENTRENAMIENTO Y TEST
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Probabilidades para AUC
            y_proba = None
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculamos métricas finales usando la función
            metrics = mt.summarize_classification(y_test, y_pred, y_proba)
            
            # Registramos cada métrica en MLflow
            for metric_name, value in metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # D. REGISTRO DEL MODELO (ARTEFACTO)
            # Esto guarda el archivo físico del modelo para usarlo después
            mlflow.sklearn.log_model(model, "model")
            
            print(f" Log completado. Test AUC: {metrics.get('auc', 0):.4f}")

if __name__ == "__main__":
    # Configuración de rutas
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_archivo = os.path.join(ruta_base, "Base_de_datos.xlsx")
    
    try:
        # Cargar datos
        df = ft.load_data(ruta_archivo)
        target = 'Pago_atiempo'
        
        # Ejecutar el tracking
        #run_experiment("Proyecto_Credito_Avance3", df, target)
        # Usamos V2 para indicar que es el modelo corregido sin leakage
        run_experiment("Proyecto_Credito_V2_NoLeakage", df, target)
        
        print("\n Experimento registrado exitosamente.")
        print("Ejecuta 'mlflow ui' en la terminal") # recordatorio :)
        
    except Exception as e:
        print(f"\n Error: {e}")