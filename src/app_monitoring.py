# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# src/app_monitoring.py

# 1. IMPORTACIÓN DE LIBRERÍAS
import streamlit as st          # ¡La estrella! Convierte el script en una página web.
import pandas as pd             # Para manejar las tablas.
import numpy as np              # Para matemáticas.
import plotly.express as px     # Para gráficos fáciles e interactivos.
import plotly.graph_objects as go # Para gráficos avanzados (personalizados al detalle).
import os                       # Para rutas de archivos.
# Importamos funciones estadísticas necesarias para comparar distribuciones.
from scipy.stats import ks_2samp, chi2_contingency 
from scipy.spatial.distance import jensenshannon # Medida de distancia entre probabilidades.

# Reutilizamos tu script de ingeniería para cargar los datos igual que antes.
import ft_engineering as ft

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 2. FUNCIONES MATEMÁTICAS (El motor de cálculo)

def calculate_psi(expected, actual, buckettype='bins', buckets=10, axis=0):
    """
    PSI (Population Stability Index): El termómetro principal.
    Divide los datos en cajitas (buckets) y compara qué % cae en cada caja
    ayer vs hoy.
    """
    def scale_range (input, min, max):
        input += -(np.min(input))
        input /= np.max(input) / (max - min)
        input += min
        return input

    # Define los puntos de corte para las cajitas
    breakpoints = np.arange(0, buckets + 1) / (buckets) * 100

    if buckettype == 'bins':
        breakpoints = scale_range(breakpoints, np.min(expected), np.max(expected))
    elif buckettype == 'quantiles':
        breakpoints = np.stack([np.percentile(expected, b) for b in breakpoints])

    # Calcula histogramas (frecuencias)
    expected_percents = np.histogram(expected, breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, breakpoints)[0] / len(actual)

    def sub_psi(e_perc, a_perc):
        # Truco técnico: Evitar dividir por cero agregando un valor minúsculo
        if a_perc == 0: a_perc = 0.0001
        if e_perc == 0: e_perc = 0.0001
        
        value = (e_perc - a_perc) * np.log(e_perc / a_perc)
        return(value)

    # Suma las diferencias de todas las cajitas
    psi_value = np.sum([sub_psi(expected_percents[i], actual_percents[i]) for i in range(0, len(expected_percents))])
    return psi_value

def calculate_ks_test(data1, data2):
    """
    Test KS: Compara la "forma" de dos montañas de datos.
    Retorna estadístico y p-value.
    """
    return ks_2samp(data1, data2)

def calculate_chi2(data1, data2):
    """
    Chi-Cuadrado: Para variables de texto (Categorías).
    ¿Han cambiado las proporciones de grupos (ej: hombres/mujeres)?
    """
    val_counts1 = data1.value_counts(normalize=True)
    val_counts2 = data2.value_counts(normalize=True)
    
    combined = pd.DataFrame({'ref': val_counts1, 'curr': val_counts2}).fillna(0)
    
    # Ajuste para que la función matemática funcione (requiere conteos enteros, no %)
    obs = np.array([combined['ref'] * 1000, combined['curr'] * 1000])
    stat, p, dof, expected = chi2_contingency(obs)
    return p

def calculate_jsd(expected, actual, buckets=10):
    """
    Divergencia Jensen-Shannon (JSD): 
    Es una forma moderna de medir distancia entre dos distribuciones.
    - 0.0: Son idénticas.
    - 1.0: Son totalmente opuestas.
    """
    # Creamos un rango fijo basado en los datos originales para comparar peras con peras
    base_min, base_max = np.min(expected), np.max(expected)
    bins = np.linspace(base_min, base_max, buckets + 1)
    
    # density=True nos da probabilidades (suma 1)
    p, _ = np.histogram(expected, bins=bins, density=True)
    q, _ = np.histogram(actual, bins=bins, density=True)
    
    # Sumamos un epsilon (1e-10) para que logaritmo no de error si hay un cero
    p = p + 1e-10
    q = q + 1e-10
    
    return jensenshannon(p, q)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 3. CONFIGURACIÓN DE LA PÁGINA WEB

# Título de la pestaña del navegador y layout ancho
st.set_page_config(page_title="Monitor de Drift", layout="wide", page_icon="🕵️")

st.title("Dashboard de Monitoreo de Data Drift")
st.markdown("""
Este tablero es el centro de control. Aquí comparamos el **Pasado (Referencia)** contra el **Presente (Actual)** para detectar si el mercado ha cambiado.
""")

# SESSION STATE (Memoria de Sesión):
# Streamlit recarga la página cada vez que haces clic. 
# Esto sirve para "recordar" los cálculos y no empezar de cero en cada clic.
if 'drift_results' not in st.session_state:
    st.session_state['drift_results'] = None

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 4. CARGA DE DATOS

# @st.cache_data: ¡Importante!
# Guarda el resultado en memoria caché. Si el archivo no cambia, no lo vuelve a leer.
# Hace que la web sea mucho más rápida.
@st.cache_data 
def get_data():
    ruta_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_archivo = os.path.join(ruta_base, "Base_de_datos.xlsx")
    df = ft.load_data(ruta_archivo)
    return df

try:
    df = get_data()
    
    # --- SIMULACIÓN DE ESCENARIO ---
    # Como solo tenemos un archivo, lo partimos a la mitad para jugar.
    
    # Las primeras 4000 filas son "Lo que el modelo aprendió"
    df_ref = df.iloc[:4000].copy() 
    
    # Las siguientes filas son "Los datos nuevos que llegan hoy"
    df_curr = df.iloc[4000:].copy()
    
    # --- SABOTAJE INTENCIONAL (Para probar la alerta) ---
    # Multiplicamos el salario por 3.5 en los datos nuevos.
    # Esto debería disparar una alarma ROJA en el dashboard.
    if 'salario_cliente' in df_curr.columns:
        df_curr['salario_cliente'] = df_curr['salario_cliente'] * 3.5 
        
except Exception as e:
    # Muestra un cuadro rojo de error en la web si falla la carga
    st.error(f"Error cargando datos: {e}")
    st.stop()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 5. BARRA LATERAL (Menú de la izquierda)

with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Slider: Permite al usuario elegir qué tan estricta es la vigilancia.
    umbral_psi = st.slider(
        "Sensibilidad de Alerta (PSI)", 
        min_value=0.1, max_value=0.5, value=0.25, step=0.01,
        help="Nivel a partir del cual consideramos que el modelo está en peligro."
    )
    
    # Leyenda de colores
    st.caption("🔴 Rojo: Drift Crítico")
    st.caption("🟡 Amarillo: Alerta Leve")
    st.caption("🟢 Verde: Estable")
    
    # Botón Principal
    if st.button("🔄 Ejecutar Análisis Completo", type="primary"):
        # Activa la señal para correr los cálculos
        st.session_state['run_analysis'] = True

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 6. LÓGICA DE PROCESAMIENTO

# Solo entramos aquí si el usuario pulsó el botón "Ejecutar"
if st.session_state.get('run_analysis', False):
    
    with st.spinner("Calculando métricas matemáticas..."): # Muestra un círculo de carga
        
        # Definimos variables a analizar
        num_cols = [
            'salario_cliente', 'capital_prestado', 'plazo_meses', 'edad_cliente', 
            'puntaje', 'total_otros_prestamos'
        ]
        cat_cols = ['tipo_laboral', 'tendencia_ingresos']
        
        # Filtro de seguridad: solo usar columnas que realmente existan
        num_cols = [c for c in num_cols if c in df_ref.columns]
        cat_cols = [c for c in cat_cols if c in df_ref.columns]
        
        results = []

        # A. Cálculos para Numéricas
        for col in num_cols:
            psi = calculate_psi(df_ref[col], df_curr[col], buckettype='quantiles')
            stat, p_value = calculate_ks_test(df_ref[col], df_curr[col])
            jsd_val = calculate_jsd(df_ref[col], df_curr[col]) # Divergencia JSD
            
            results.append({
                "Variable": col,
                "Tipo": "Numérica",
                "Métrica 1": psi,
                "Nombre M1": "PSI",
                "Métrica 2": f"KS p={p_value:.3f} | JSD={jsd_val:.3f}",
            })

        # B. Cálculos para Categóricas
        for col in cat_cols:
            # Convertir a texto siempre para evitar fallos
            ref_c = df_ref[col].astype(str)
            cur_c = df_curr[col].astype(str)
            p_val = calculate_chi2(ref_c, cur_c)
            
            results.append({
                "Variable": col,
                "Tipo": "Categórica",
                "Métrica 1": p_val,
                "Nombre M1": "Chi2 p-val",
                "Métrica 2": "-",
            })
            
        # Guardamos todo en la "Memoria" (Session State)
        st.session_state['drift_results'] = pd.DataFrame(results)
        st.session_state['run_analysis'] = False # Apagamos el botón para no recalcular sin querer
        st.rerun() # Refrescamos la pantalla para mostrar los resultados nuevos

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# 7. VISUALIZACIÓN (La parte bonita)

# Si ya tenemos resultados en memoria, los mostramos
if st.session_state['drift_results'] is not None:
    
    drift_df = st.session_state['drift_results'].copy()
    
    # Función dinámica para asignar colores según el slider que mueve el usuario
    def assign_status(row):
        val = row['Métrica 1']
        if row['Tipo'] == 'Numérica': 
            # Reglas para PSI
            if val < 0.1: return "🟢 Estable"
            elif val < umbral_psi: return "🟡 Alerta" # Usa el valor del slider
            else: return "🔴 Drift Crítico"
        else: 
            # Reglas para Chi2
            if val > 0.05: return "🟢 Estable"
            else: return "🔴 Drift (Significativo)"

    # Aplicamos la función fila por fila
    drift_df['Estado'] = drift_df.apply(assign_status, axis=1)
    
    # --- TARJETAS DE RESUMEN (KPIs) ---
    col1, col2, col3 = st.columns(3)
    n_critico = len(drift_df[drift_df['Estado'].str.contains("🔴")])
    n_alerta = len(drift_df[drift_df['Estado'].str.contains("🟡")])
    
    # Mostramos números grandes arriba
    col1.metric("Variables Analizadas", len(drift_df))
    col2.metric("Alertas Leves", n_alerta)
    col3.metric("🚨 Variables Críticas", n_critico, delta_color="inverse")
    
    # --- TABLA DE DATOS ---
    st.subheader("📋 Reporte Detallado")
    
    # Función de estilo para pintar las letras de la tabla
    def highlight_drift(val):
        color = 'red' if '🔴' in val else 'orange' if '🟡' in val else 'green'
        return f'color: {color}; font-weight: bold'

    # Mostramos la tabla interactiva
    st.dataframe(
        drift_df[['Variable', 'Tipo', 'Nombre M1', 'Métrica 1', 'Métrica 2', 'Estado']]
        .style.map(highlight_drift, subset=['Estado']) # Aplica colores a la columna Estado
        .format({'Métrica 1': '{:.4f}'}), # Redondea a 4 decimales
        use_container_width=True 
    )
    
    st.divider() # Línea separadora visual
    
    # --- INSPECTOR DE GRÁFICOS ---
    st.subheader("🔎 Inspector Visual")
    st.info("Selecciona una variable para ver cómo cambió su forma:")
    
    # Menú desplegable para elegir qué gráfico ver
    lista_vars = drift_df['Variable'].tolist()
    selected_var = st.selectbox("Variable:", lista_vars)
    
    # Creamos dos columnas: una ancha para el gráfico, otra angosta para detalles
    col_graph1, col_graph2 = st.columns([2, 1])
    
    with col_graph1:
        # Lógica para dibujar el gráfico correcto (Histograma o Barras)
        if selected_var in df_ref.select_dtypes(include=np.number).columns:
            # Gráfico Plotly interactivo (puedes hacer zoom)
            fig = go.Figure()
            # Histograma Azul = Pasado
            fig.add_trace(go.Histogram(x=df_ref[selected_var], name='Referencia', opacity=0.6, marker_color='blue'))
            # Histograma Rojo = Presente
            fig.add_trace(go.Histogram(x=df_curr[selected_var], name='Actual', opacity=0.6, marker_color='red'))
            fig.update_layout(barmode='overlay', title=f"Distribución: {selected_var}")
            st.plotly_chart(fig, use_container_width=True) 
        else:
            # Gráfico de Barras para texto (ej: Tipo Laboral)
            ref_vc = df_ref[selected_var].value_counts(normalize=True).reset_index()
            ref_vc.columns = ['Valor', 'Proporcion']
            ref_vc['Dataset'] = 'Referencia'
            curr_vc = df_curr[selected_var].value_counts(normalize=True).reset_index()
            curr_vc.columns = ['Valor', 'Proporcion']
            curr_vc['Dataset'] = 'Actual'
            comp_df = pd.concat([ref_vc, curr_vc])
            
            fig = px.bar(comp_df, x='Valor', y='Proporcion', color='Dataset', barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
    with col_graph2:
        # Panel de información a la derecha del gráfico
        row = drift_df[drift_df['Variable'] == selected_var].iloc[0]
        st.info(f"**Estado:** {row['Estado']}")
        st.write(f"**{row['Nombre M1']}:** {row['Métrica 1']:.4f}")
        
        # Mensaje de ayuda si hay error crítico
        if "🔴" in row['Estado']:
            st.error("Detectamos un cambio fuerte. Revisa si hubo cambios en el mercado o errores de datos.")

else:
    # Pantalla de bienvenida antes de que el usuario pulse nada
    st.info("Ejecutar Análisis Completo en la barra lateral para comenzar.")