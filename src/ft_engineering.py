# src/ft_engineering.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer

def load_data(filepath):
    if filepath.endswith('.xlsx'):
        print(f"Cargando archivo Excel: {filepath}")
        df = pd.read_excel(filepath, engine='openpyxl')
    else:
        print(f"Cargando archivo CSV: {filepath}")
        df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    return df

def log_transform(X):
    return np.log1p(X.astype(float))

def get_preprocessor(num_norm_features, num_log_features, cat_features):
    num_norm_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()) 
    ])

    num_log_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('log', FunctionTransformer(log_transform, validate=False)),
        ('scaler', StandardScaler()) 
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_norm', num_norm_transformer, num_norm_features),
            ('num_log', num_log_transformer, num_log_features),
            ('cat', cat_transformer, cat_features)
        ],
        remainder='drop' 
    )
    return preprocessor

def perform_engineering(df, target_col):
    # 1. DEFINICIÓN DE COLUMNAS (SIN DATA LEAKAGE)
    
    # HEMOS QUITADO: 'saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal'
    # Estas variables afectaban la respuesta. Nos quedamos con lo que sabemos ANTES del préstamo.
    
    num_log_cols = [
        'capital_prestado', 
        'salario_cliente', 
        'total_otros_prestamos', 
        'cuota_pactada', 
        'promedio_ingresos_datacredito'
    ] 
    
    num_norm_cols = [
        'plazo_meses', 
        'edad_cliente', 
        # 'puntaje', 'puntaje_datacredito', 
        'cant_creditosvigentes', 'creditos_sectorFinanciero', 
        'creditos_sectorCooperativo', 'creditos_sectorReal'
    ]

    cat_cols = ['tipo_laboral', 'tendencia_ingresos'] 

    # 2. SEPARACIÓN X / y
    # Agregamos las columnas de leakage a la lista de eliminación para estar seguros
    cols_to_drop = [
        'fecha_prestamo', target_col, 'huella_consulta', 'tipo_credito',
        'saldo_mora', 'saldo_mora_codeudor', 'saldo_total', 'saldo_principal'
    ]
    
    # Filtramos para borrar solo las que existan en el df
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[target_col]

    # 3. CORRECCIÓN DE TIPOS
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype(str)
            X.loc[X[col] == 'nan', col] = np.nan

    for col in num_log_cols + num_norm_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # 4. SPLIT TRAIN/TEST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. TRANSFORMACIÓN
    preprocessor = get_preprocessor(num_norm_cols, num_log_cols, cat_cols)
    
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # 6. DIAGNÓSTICO DE COLUMNAS USADAS
    print("\n DIAGNÓSTICO DE COLUMNAS USADAS:")
    print(f"Total columns: {len(num_norm_cols) + len(num_log_cols) + len(cat_cols)}")
    print("Numéricas Normales:", num_norm_cols)
    print("Numéricas Log:", num_log_cols)
    print("Categoricas:", cat_cols)
    print("-----------------------------------\n")
    
    return X_train_processed, X_test_processed, y_train, y_test, preprocessor

if __name__ == "__main__":
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
            X_tr, X_te, y_tr, y_te, prep = perform_engineering(df, target_col=target)
            print("\n ¡ÉXITO! Ingeniería sin Data Leakage completada.")
            print(f"Dimensiones Train: {X_tr.shape}") # Nota: deberían ser menos columnas que antes (aprox 56)
            
    except Exception as e:
        print(f"\n Error: {e}")