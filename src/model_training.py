# model_training.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Modelos y Métricas
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import cross_validate

# --- IMPORTANTE: Importamos el script de ingeniería ---
import ft_engineering as ft

def build_models():
    """
    Define los modelos candidatos a evaluar con hiperparámetros base.
    """
    models = [
        # Modelo base simple (lineal)
        ('LogisticReg', LogisticRegression(max_iter=1000, random_state=42)),
        
        # Modelos basados en árboles (robustos a outliers y no lineales)
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('GradientBoost', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    return models

def summarize_classification(y_true, y_pred, y_proba=None):
    """
    Calcula un diccionario con las métricas clave de evaluación.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Calcular AUC si tenemos probabilidades disponibles
    if y_proba is not None:
        try:
            # Para binario asumimos que y_proba es la prob de la clase positiva
            metrics['auc'] = roc_auc_score(y_true, y_proba)
        except ValueError:
            metrics['auc'] = 0.0
            
    return metrics

def train_and_eval(X_train, y_train, X_test, y_test):
    """
    Función maestra: Entrena, valida (CV) y evalúa en test.
    """
    models = build_models()
    results_list = []
    
    print(f"Iniciando entrenamiento de {len(models)} modelos...\n")

    for name, model in models:
        print(f"--> Procesando: {name}...")
        
        # 1. EVALUACIÓN DE CONSISTENCIA (Validación Cruzada en Train)
        # Esto nos dice qué tan estable es el modelo (Escalabilidad)
        cv_results = cross_validate(model, X_train, y_train, cv=5, scoring='accuracy')
        cv_mean = cv_results['test_score'].mean()
        cv_std = cv_results['test_score'].std()
        
        # 2. ENTRENAMIENTO FINAL
        model.fit(X_train, y_train)
        
        # 3. EVALUACIÓN DE PERFORMANCE (Predicción en Test)
        y_pred = model.predict(X_test)
        
        # Obtener probabilidades para AUC (si el modelo lo soporta)
        y_proba = None
        if hasattr(model, "predict_proba"):
            # Tomamos la columna 1 (probabilidad de clase positiva/default)
            y_proba = model.predict_proba(X_test)[:, 1]
            
        # Calcular métricas finales
        metrics = summarize_classification(y_test, y_pred, y_proba)
        
        # Consolidar resultados
        results_list.append({
            'Model': name,
            'CV_Accuracy_Mean': cv_mean,
            'CV_Accuracy_Std': cv_std, # <--- Clave para ver estabilidad
            'Test_Accuracy': metrics['accuracy'],
            'Test_F1': metrics['f1_score'],
            'Test_AUC': metrics.get('auc', 0.0),
            'Test_Recall': metrics['recall']
        })

    return pd.DataFrame(results_list)

def plot_comparison(results_df):
    """
    Genera un gráfico de barras comparativo.
    """
    # Preparamos datos para Seaborn (formato largo)
    metrics_to_plot = ["Test_AUC", "Test_F1", "CV_Accuracy_Mean"]
    melted_df = results_df.melt(id_vars="Model", value_vars=metrics_to_plot, var_name="Metric", value_name="Score")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted_df, x="Model", y="Score", hue="Metric", palette="viridis")
    
    plt.title("Comparación de Modelos: Performance vs Estabilidad")
    plt.ylim(0, 1.1)
    plt.ylabel("Score (0-1)")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. PREPARACIÓN DE DATOS (Usando ft_engineering)
    # Buscamos el archivo usando la misma lógica que ya funcionó
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_archivo = os.path.join(ruta_base, "Base_de_datos.xlsx")
    
    try:
        print("1. Cargando y procesando datos...")
        df = ft.load_data(ruta_archivo)
        target = 'Pago_atiempo' 
        
        # Ejecutamos el pipeline de ingeniería
        X_train, X_test, y_train, y_test, _ = ft.perform_engineering(df, target_col=target)
        
        # 2. ENTRENAMIENTO Y SELECCIÓN
        print("\n2. Entrenando modelos...")
        results_df = train_and_eval(X_train, y_train, X_test, y_test)
        
        # 3. REPORTE DE RESULTADOS
        print("\n" + "="*50)
        print("TABLA RESUMEN DE EVALUACIÓN")
        print("="*50)
        # Ordenamos por AUC (mejor capacidad de distinción)
        print(results_df.sort_values(by="Test_AUC", ascending=False).to_string(index=False))
        
        # 4. SELECCIÓN DEL MEJOR MODELO
        # Criterio: Mejor balance entre AUC y Estabilidad (baja desviación std)
        best_model_row = results_df.sort_values(by="Test_AUC", ascending=False).iloc[0]
        print(f"\n🏆 El modelo recomendado es: {best_model_row['Model']}")
        print(f"   AUC: {best_model_row['Test_AUC']:.4f}")
        print(f"   Estabilidad (CV Std): {best_model_row['CV_Accuracy_Std']:.4f}")
        
        # 5. GRÁFICOS
        print("\nGenerando gráficos...")
        plot_comparison(results_df)
        
    except Exception as e:
        print(f"\n Error en el proceso: {e}")