import os
import pandas as pd


def cargarDatos():
    # Ruta del direcctorio donde esta el archivo en la carpeta
    ruta_actual = os.path.dirname(os.path.abspath(__file__))

    # Subir un nivel de carpetas para llegar a la carpeta donde esta la base de datos
    ruta_proyecto = os.path.dirname(ruta_actual)

    # Ruta completa a la base de datos
    ruta_excel = os.path.join(ruta_proyecto, "Base_de_datos.xlsx")

    #Leer datos e imprimir
    df = pd.read_excel(ruta_excel)
    print(df)

    return df

if __name__ == "__main__":
    datos = cargarDatos()
    print(datos.head())
    print(datos.columns)