# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# src/ft_engineering.py

# 1. IMPORTACIÓN DE LIBRERÍAS
# Pandas y Numpy: Para manipular tablas y operaciones matemáticas.
import pandas as pd
import numpy as np
import os
# Sklearn: Herramientas para preparar datos para Machine Learning.
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Para rellenar datos faltantes
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer # Para aplicar reglas distintas a columnas distintas

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def load_data(filepath):
    """
    Función utilitaria para leer el archivo de datos, ya sea Excel o CSV.
    """
    if filepath.endswith('.xlsx'):
        print(f"Cargando archivo Excel: {filepath}")
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        print(f"Cargando archivo CSV: {filepath}")
        df = pd.read_csv(filepath)
    
    # Limpieza básica: Quita espacios en blanco de los nombres de las columnas
    # Ej: " Salario " se convierte en "Salario"
    df.columns = df.columns.str.strip()
    return df

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def log_transform(X):
    """
    Transformación matemática (Logaritmo).
    Sirve para 'comprimir' números muy grandes (como salarios millonarios)
    y hacerlos más manejables para el modelo.
    """
    # np.log1p calcula log(1 + x) para evitar errores si el valor es 0.
    return np.log1p(X.astype(float))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def get_preprocessor(num_norm_features, num_log_features, cat_features):
    """
    Esta función crea la 'receta' de limpieza. Define qué hacer con cada tipo de columna.
    """
    
    # 1. Tubería para números normales (ej: edad, plazo)
    num_norm_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')), # Si falta un dato, pon el promedio.
        ('scaler', StandardScaler()) # Pone los datos en una escala común (ej: -1 a 1).
    ])

    # 2. Tubería para números con rangos muy amplios (ej: salarios)
    num_log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('log', FunctionTransformer(log_transform, validate=False)), # Aplica logaritmo.
        ('scaler', StandardScaler())
    ])

    # 3. Tubería para texto/categorías (ej: Tipo de contrato)
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')), # Si falta, pon "MISSING".
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Convierte texto a columnas de 0s y 1s.
    ])

    # Unimos las 3 tuberías en un solo procesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num_norm', num_norm_transformer, num_norm_features),
            ('num_log', num_log_transformer, num_log_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop' # ¡Importante! Borra cualquier columna que no hayamos mencionado arriba.
    )
    return preprocessor

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def perform_engineering(df, target_col):
    """
    Función Maestra de Ingeniería:
    Selecciona columnas, limpia, divide en Train/Test y transforma los datos.
    """
    
    # 1. DEFINICIÓN DE COLUMNAS (SELECCIÓN DE VARIABLES)
    
    # PREVENCIÓN DE DATA LEAKAGE (Fuga de datos):
    # Quitamos columnas como 'saldo_mora' porque eso pasa DESPUÉS de dar el préstamo.
    # El modelo solo debe conocer lo que sabíamos el día que el cliente pidió el dinero.
    
    # Variables numéricas grandes (se les aplicará logaritmo)
    num_log_cols = [
        'capital_prestado', 
        'salario_cliente', 
        'total_otros_prestamos', 
        'cuota_pactada', 
        'promedio_ingresos_datacredito'
    ] 
    
    # Variables numéricas estándar
    num_norm_cols = [
        'plazo_meses', 
        'edad_cliente', 
        'cant_creditosvigentes', 'creditos_sectorFinanciero', 
        'creditos_sectorCooperativo', 'creditos_sectorReal'
    ]

    # Variables de texto (categóricas)
    cat_cols = ['tipo_laboral', 'tendencia_ingresos'] 

    # 2. LIMPIEZA INICIAL
    # Definimos qué borrar (incluyendo la variable objetivo para que no esté en X)
    cols_to_drop = [
        'fecha_prestamo', target_col, 'huella_consulta', 'tipo_credito',
        'saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal'
    ]
    
    # Validamos que las columnas a borrar existan antes de intentar borrarlas
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    # Separamos X (variables explicativas) e y (lo que queremos predecir)
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # 3. CORRECCIÓN DE TIPOS DE DATOS
    # Aseguramos que el texto sea texto y los números sean números
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str) # Forzar a string
            # Convertir la palabra 'nan' literal en un valor vacío real de numpy
            X.loc[X[col] == 'nan', col] = np.nan

    for col in num_log_cols + num_norm_cols:
        if col in X.columns:
            # Convertir a número, si falla pone NaN (Not a Number)
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # 4. DIVISIÓN TRAIN / TEST (Entrenamiento y Prueba)
    # 80% para entrenar, 20% para probar.
    # stratify=y asegura que haya la misma proporción de "buenos/malos" pagadores en ambos grupos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. EJECUCIÓN DE LA TRANSFORMACIÓN
    # Traemos la receta que creamos arriba
    preprocessor = get_preprocessor(num_norm_cols, num_log_cols, cat_cols)
    
    # fit_transform en Train: "Aprende" los promedios y transforma los datos.
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # transform en Test: Usa los promedios aprendidos en Train para transformar Test.
    # (Nunca hacemos fit en Test para no hacer trampa).
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. DIAGNÓSTICO (Para que el usuario vea qué pasó)
    print("\n DIAGNÓSTICO DE COLUMNAS USADAS:")
    print(f"Total variables de entrada: {len(num_norm_cols) + len(num_log_cols) + len(cat_cols)}")
    print("Numéricas Normales:", num_norm_cols)
    print("Numéricas Log:", num_log_cols)
    print("Categoricas:", cat_cols)
    print("-----------------------------------\n")
    
    # Devolvemos los datos listos y el preprocesador (por si queremos usarlo con nuevos clientes en el futuro)
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

# --- BLOQUE DE PRUEBA ---
if __name__ == "__main__":
    # Configuración de rutas para encontrar el archivo
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nombre_archivo = "Base_de_datos.xlsx"
    ruta_archivo = os.path.join(ruta_base, nombre_archivo)
    
    try:
        print(f"Buscando en: {ruta_archivo}")
        df = load_data(ruta_archivo)
        target = 'Pago_atiempo'
        
        if target not in df.columns:
            print(f" Error: La columna objetivo '{target}' no está en el DataFrame.")
        else:
            # Probamos la función principal
            X_tr, X_te, y_tr, y_te, prep = perform_engineering(df, target_col=target)
            print("\n ¡ÉXITO! Ingeniería sin Data Leakage completada.")
            # .shape nos dice cuántas filas y columnas quedaron (filas, columnas)
            print(f"Dimensiones Train: {X_tr.shape}") 
            
    except Exception as e:
        print(f"\n Error: {e}")