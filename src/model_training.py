# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# model_training.py

# 1. IMPORTACIÓN DE LIBRERÍAS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# 2. IMPORTACIÓN DE HERRAMIENTAS DE MACHINE LEARNING (Scikit-Learn)
# Importamos los "algoritmos" que aprenderán de los datos.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# Importamos las "reglas" para medir qué tan bien aprendieron (métricas).
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
# Importamos herramientas para validar el entrenamiento.
from sklearn.model_selection import cross_validate

# --- IMPORTANTE: Importamos nuestro propio script de limpieza ---
# Este 'ft_engineering' es un archivo .py que debe estar en la misma carpeta.
# Contiene las funciones para limpiar y preparar los datos antes de este paso.

import ft_engineering as ft

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def build_models():
    """
    Función que crea una lista de los modelos (algoritmos) que queremos poner a competir.
    """
    models = [
        # 1. Regresión Logística: Es el modelo base. Rápido y simple.
        # Sirve para ver si el problema se puede resolver con una línea recta.
        ('LogisticReg', LogisticRegression(max_iter=1000, random_state=42)),
        # 2. Random Forest: Crea muchos "árboles de decisión" y votan entre ellos.
        # Es muy bueno manejando datos complejos y valores atípicos.
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        # 3. Gradient Boosting: Similar a Random Forest, pero cada árbol corrige los errores del anterior.
        # Suele ser el más preciso, pero tarda más en entrenar.
        ('GradientBoost', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    return models

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -¿
def summarize_classification(y_true, y_pred, y_proba=None):
    """
    Esta función actúa como el 'profesor' que califica el examen del modelo.
    Recibe las respuestas correctas (y_true) y las predicciones del modelo (y_pred).
    """
    metrics = {
        # Accuracy: ¿Qué porcentaje total acertó?
        'accuracy': accuracy_score(y_true, y_pred),
        # F1_score: Un balance entre precisión y exhaustividad (útil si hay desbalance de datos).
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        # Precision: De los que dijo que eran positivos, ¿cuántos lo eran realmente?
        'precision': precision_score(y_true, y_pred, average='weighted'),
        # Recall: De todos los que eran positivos en la realidad, ¿cuántos encontró?
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    
    # AUC (Área bajo la curva): Mide qué tan bien el modelo distingue entre clases.
    # Requiere probabilidades (y_proba), no solo respuestas si/no.
    if y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            # Si falla el cálculo, asumimos 0.
            metrics['auc'] = 0.0
    return metrics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def train_and_eval(X_train, y_train, X_test, y_test):
    """
    Función Maestra: Aquí ocurre la magia.
    Recibe los datos de entrenamiento (Train) y los de prueba (Test).
    """
    # Obtenemos la lista de competidores (los modelos definidos arriba)
    models = build_models()
    results_list = [] # Aquí guardaremos las calificaciones de cada uno
    
    print(f"Iniciando entrenamiento de {len(models)} modelos...\n")

    # Bucle: Pasamos por cada modelo uno por uno
    for name, model in models:
        print(f"--> Procesando: {name}...")
        
        # PASO 1: Validación Cruzada (Cross-Validation)
        # Esto divide los datos de entrenamiento en 5 partes. Entrena en 4 y prueba en 1, repetidamente.
        # Sirve para ver si el modelo es "estable" o si tuvo suerte con los datos.
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_results['test_score'].mean() # Promedio de nota
        cv_std = cv_results['test_score'].std()   # Desviación (¿Qué tanto varió su nota?)
        
        # PASO 2: Entrenamiento Final
        # Ahora sí, le enseñamos con TODOS los datos de entrenamiento.
        model.fit(X_train, y_train)
        
        # PASO 3: Examen Final (Predicción)
        # Le pedimos que prediga los resultados de los datos de prueba (que nunca ha visto).
        y_pred = model.predict(X_test)
        
        # Intentamos obtener la probabilidad (ej: 80% seguro de que sí, 20% que no)
        y_proba = None
        if hasattr(model, "predict_proba"):
            # Nos quedamos con la probabilidad de la clase positiva (columna 1)
            y_proba = model.predict_proba(X_test)[:, 1]
            
        # Calculamos las notas finales usando la función que creamos antes
        metrics = summarize_classification(y_test, y_pred, y_proba)
        
        # Guardamos todo en nuestra lista de resultados
        results_list.append({
            'Model': name,
            'CV_Accuracy_Mean': cv_mean, # Nota promedio en prácticas
            'CV_Accuracy_Std': cv_std,   # Estabilidad
            'Test_Accuracy': metrics['accuracy'],
            'Test_F1': metrics['f1_score'],
            'Test_AUC': metrics.get('auc', 0.0), # Capacidad de distinción
            'Test_Recall': metrics['recall']
        })

    # Convertimos la lista en una tabla bonita de Pandas y la devolvemos
    return pd.DataFrame(results_list)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def plot_comparison(results_df):
    """
    Crea un gráfico de barras para comparar visualmente los modelos.
    """
    # Transformamos la tabla para que Seaborn pueda graficarla fácil (formato largo)
    metrics_to_plot = ["Test_AUC", "Test_F1", "CV_Accuracy_Mean"]
    melted_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    # Configuración del gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    # Etiquetas y título
    plt.title("Comparación de Modelos: Performance vs Estabilidad")
    plt.ylim(0, 1.1) # Eje Y va de 0 a 1.1 para que se vea bien
    plt.ylabel("Score (0-1)")
    plt.grid(axis='y', linestyle='--', alpha=0.3) # Rejilla de fondo
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Leyenda fuera del gráfico
    plt.tight_layout() # Ajustar márgenes
    plt.show() # Mostrar ventana

# --- BLOQUE PRINCIPAL ---
# Este bloque solo se ejecuta si corremos este archivo directamente.
if __name__ == "__main__":
    
    # 1. PREPARACIÓN DE RUTAS
    # Busca dónde está este archivo en tu computadora para encontrar la base de datos
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_archivo = os.path.join(ruta_base, "Base_de_datos.xlsx")
    
    try:
        print("1. Cargando y procesando datos...")
        # Llama a la función 'load_data' del archivo ft_engineering importado como 'ft'
        df = ft.load_data(ruta_archivo)
        target = 'Pago_atiempo' # Esta es la columna que queremos predecir
        
        # Llama a la ingeniería de características (limpieza y transformación)
        # Nos devuelve los datos ya divididos en X (variables) e y (objetivo) para entrenar y probar
        X_train, X_test, y_train, y_test, _ = ft.perform_engineering(df, target_col=target)
        
        # 2. ENTRENAMIENTO
        print("\n2. Entrenando modelos...")
        results_df = train_and_eval(X_train, y_train, X_test, y_test)
        
        # 3. RESULTADOS EN CONSOLA
        print("\n" + "="*50)
        print("TABLA RESUMEN DE EVALUACIÓN")
        print("="*50)
        # Imprime la tabla ordenando los mejores modelos arriba (según AUC)
        print(results_df.sort_values(by="Test_AUC", ascending=False).to_string(index=False))
        
        # 4. RECOMENDACIÓN AUTOMÁTICA
        # Elige automáticamente el mejor modelo basándose en la métrica AUC
        best_model_row = results_df.sort_values(by="Test_AUC", ascending=False).iloc[0]
        print(f"\n🏆 El modelo recomendado es: {best_model_row['Model']}")
        print(f"   AUC: {best_model_row['Test_AUC']:.4f}")
        print(f"   Estabilidad (CV Std): {best_model_row['CV_Accuracy_Std']:.4f}")
        
        # 5. RESULTADOS GRÁFICOS
        print("\nGenerando gráficos...")
        plot_comparison(results_df)
        
    except Exception as e:
        # Si algo falla (ej: no encuentra el archivo), imprime el error amigablemente
        print(f"\n Error en el proceso: {e}")