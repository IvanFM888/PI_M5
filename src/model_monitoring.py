# src/model_monitoring.py

# 1. IMPORTACIÓN DE LIBRERÍAS
# Numpy y Pandas: Para manejo de datos y matemáticas.
import numpy as np
import pandas as pd
# Scipy Stats: Librería de estadística avanzada.
# Traemos pruebas específicas: KS, Chi-cuadrado y Entropía.
from scipy.stats import ks_2samp, chi2_contingency, entropy

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    Calcula el PSI (Índice de Estabilidad Poblacional).
    Es la métrica ESTÁNDAR en la industria para saber si una variable numérica cambió.
    
    Regla de oro:
    - PSI < 0.1: Todo bien, nada ha cambiado (Verde).
    - PSI 0.1 - 0.25: Cambio moderado, ten cuidado (Amarillo).
    - PSI > 0.25: Drift Crítico. El modelo ya no sirve (Rojo).
    """
    
    # Función auxiliar interna para ajustar escalas (normalizar de 0 a 1)
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    # Creamos los puntos de corte para dividir los datos en "cubetas" (buckets)
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    # Decidimos cómo cortar los rangos
    if buckettype == 'bins':
        # Cortes por rango fijo (ej: 0-10, 10-20, 20-30)
        breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    elif buckettype == 'quantiles':
        # Cortes por cantidad de gente (ej: el 10% más pobre, el siguiente 10%, etc.)
        breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

    # Calculamos qué % de datos cae en cada cubeta para la referencia (expected) y el actual
    # np.histogram cuenta cuántos datos caen en cada rango.
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    # Función para calcular la diferencia matemática en cada cubeta
    def sub_psi(e_perc, a_perc):
        # Truco: Si un % es 0, le ponemos un valor diminuto (0.0001) para evitar dividir por cero
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001

        # Fórmula matemática del PSI
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    # Sumamos las diferencias de todas las cubetas para tener el score final
    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])
    return psi_value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calculate_ks_test(ref_data, curr_data):
    """
    Test de Kolmogorov-Smirnov (KS).
    Compara la forma de dos curvas de distribución numérica.
    Retorna p-value: Si es muy bajo, es probable que los datos sean diferentes.
    """
    statistic, p_value = ks_2samp(ref_data, curr_data)
    return statistic, p_value

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def calculate_chi2(ref_data, curr_data):
    """
    Test de Chi-Cuadrado (Chi2).
    Se usa para comparar variables DE TEXTO (Categorías).
    Ej: ¿La proporción de Hombres/Mujeres cambió?
    """
    # 1. Calculamos porcentajes de cada categoría
    ref_counts = ref_data.value_counts(normalize=True)
    curr_counts = curr_data.value_counts(normalize=True)
    
    # 2. Alineamos los datos en una tabla comparativa
    # (fillna(0) es por si aparece una categoría nueva que antes no existía)
    df_comp = pd.DataFrame({'ref': ref_counts, 'curr': curr_counts}).fillna(0)
    
    # 3. Truco matemático: Chi2 necesita conteos enteros, no porcentajes.
    # Simulamos una población de 1000 personas para hacer el cálculo.
    obs = np.array([df_comp['ref'].values * 1000, df_comp['curr'].values * 1000])
    
    # Ejecutamos el test
    chi2, p, dof, ex = chi2_contingency(obs)
    return p # Retornamos la probabilidad de que sean iguales

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def check_drift(df_reference, df_current, num_cols, cat_cols):
    """
    FUNCIÓN MAESTRA: Recibe los datos viejos (reference) y nuevos (current).
    Revisa columna por columna y genera un reporte de salud.
    """
    drift_report = []

    # --- 1. Análisis de Variables Numéricas ---
    for col in num_cols:
        # Solo analizamos si la columna existe en ambos lados
        if col in df_reference.columns and col in df_current.columns:
            
            # Calculamos PSI (Estabilidad)
            psi = calculate_psi(df_reference[col], df_current[col], buckettype='quantiles')
            # Calculamos KS (Forma de distribución)
            ks_stat, ks_p = calculate_ks_test(df_reference[col], df_current[col])
            
            # --- SEMÁFORO DE ALERTA ---
            status = "OK"
            if psi >= 0.1: status = "Alerta"        # Cambio notable
            if psi >= 0.25: status = "DRIFT CRÍTICO" # Cambio peligroso
            
            # Guardamos resultado
            drift_report.append({
                "Variable": col,
                "Tipo": "Numérica",
                "Métrica Principal": f"PSI: {psi:.4f}",
                "Métrica Secundaria": f"KS p-value: {ks_p:.4f}",
                "Score": psi, # Usamos PSI para ordenar gravedad
                "Estado": status
            })

    # --- 2. Análisis de Variables Categóricas (Texto) ---
    for col in cat_cols:
        if col in df_reference.columns and col in df_current.columns:
            # Convertimos a texto (str) para evitar errores si hay números mezclados
            ref_c = df_reference[col].astype(str)
            cur_c = df_current[col].astype(str)
            
            # Calculamos Chi-Cuadrado
            p_val = calculate_chi2(ref_c, cur_c)
            
            # Alerta: Si p-value < 0.05, estadísticamente son diferentes
            status = "OK"
            if p_val < 0.05: status = "DRIFT (p<0.05)"
            
            drift_report.append({
                "Variable": col,
                "Tipo": "Categórica",
                "Métrica Principal": f"Chi2 p-val: {p_val:.4f}",
                "Métrica Secundaria": "-",
                "Score": 1 - p_val, # Invertimos para que score alto sea "más alerta"
                "Estado": status
            })
            
    # Convertimos la lista de diccionarios en un DataFrame bonito
    return pd.DataFrame(drift_report)