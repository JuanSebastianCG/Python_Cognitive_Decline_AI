# ./model/preprocessing/data_extraction.py
import pandas as pd

initial_columns_to_drop = ['columna_no_relevante_1', 'columna_no_relevante_2']

def load_data(file_path):
    """
    Carga datos desde un archivo CSV y retorna un DataFrame de pandas.
    """
    return pd.read_csv(file_path)

def prepare_data(df, columns_to_drop = initial_columns_to_drop):
    """
    Realiza las primeras limpiezas de datos, como la eliminación de columnas innecesarias.
    """
    # Ejemplo de eliminación de columnas no relevantes

    df = df.drop(columns=columns_to_drop, errors='ignore')
    return df
