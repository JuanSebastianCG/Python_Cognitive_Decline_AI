import pandas as pd
import pickle
from sklearn.model_selection import train_test_split

# Definir las columnas iniciales a eliminar, debe ser reemplazado con los nombres reales de las columnas
initial_columns_to_drop = ['column1', 'column2']  # Replace with actual column names to drop

class DataExtractor:
    def __init__(self, base_path="../../data/", columns_to_drop=initial_columns_to_drop):
        """
        Inicializa la clase DataExtractor, que se encarga de cargar, procesar y dividir conjuntos de datos
        en conjuntos de entrenamiento, validación y prueba.

        Parámetros:
        base_path (str): Ruta base donde se almacenan los datos.
        columns_to_drop (list): Lista de columnas a eliminar al cargar los datos.
        """
        self.base_path = base_path
        self.columns_to_drop = columns_to_drop

    def load_data(self, nameFolderRaw, file_name):
        """
        Carga datos desde un archivo CSV, eliminando columnas innecesarias si están presentes.

        Parámetros:
        nameFolderRaw (str): Nombre de la carpeta que contiene los datos en bruto.
        file_name (str): Nombre del archivo CSV a cargar.

        Returns:
        DataFrame: El DataFrame con los datos cargados y las columnas especificadas eliminadas.
        """
        folder_path = self.base_path + nameFolderRaw + file_name
        data = pd.read_csv(folder_path)
        data = data.drop(columns=self.columns_to_drop, errors='ignore')
        return data

    def load_data_processed(self, namefile):
        """
        Carga un archivo de datos ya procesados desde un archivo pickle.

        Parámetros:
        namefile (str): Nombre del archivo pickle a cargar.

        Returns:
        object: El objeto de datos cargado desde el archivo pickle.
        """
        with open(self.base_path + "processed/" + namefile, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def save_data_pickle(self, data, namefile, filter_condition=None):
        """
        Guarda un conjunto de datos en un archivo pickle, con una opción para aplicar un filtro antes de guardar.

        Parámetros:
        data (DataFrame or object): El conjunto de datos o el objeto a guardar.
        namefile (str): Nombre del archivo pickle donde se guardará.
        filter_condition (str, optional): Condición de filtrado a aplicar antes de guardar los datos.
        """
        if filter_condition:
            data = data.query(filter_condition)
        with open(self.base_path + "processed/" + namefile, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def extract_test_validation_training_data(data, training_ratio=0.7, test_ratio=0.2, validation_ratio=0.1):
        """
        Divide un conjunto de datos en conjuntos de entrenamiento, validación y prueba según las proporciones especificadas.

        Parámetros:
        data (DataFrame): El conjunto de datos a dividir.
        training_ratio (float): Proporción del conjunto de datos destinado al entrenamiento.
        test_ratio (float): Proporción del conjunto de datos destinado a las pruebas.
        validation_ratio (float): Proporción del conjunto de datos destinado a la validación.

        Returns:
        list: Una lista de tuplas que contienen:
            - Datos de entrenamiento (X e y)
            - Datos de validación (X e y)
            - Datos de prueba (X e y)
        """
        # Dividir datos en conjunto de entrenamiento y un conjunto temporal (test + validación)
        training_data, temp_data = train_test_split(data, test_size=1-training_ratio, random_state=42)
        # Calcular la proporción relativa para dividir el conjunto temporal en validación y prueba
        relative_validation_ratio = validation_ratio / (test_ratio + validation_ratio)
        validation_data, test_data = train_test_split(temp_data, test_size=relative_validation_ratio, random_state=42)
        
        # Separar características (X) y la variable objetivo (y) para cada conjunto
        x_training_data = training_data.drop(columns=['target'])
        y_training_data = training_data['target']
        
        x_validation_data = validation_data.drop(columns=['target'])
        y_validation_data = validation_data['target']
        
        x_test_data = test_data.drop(columns=['target'])
        y_test_data = test_data['target']
        
        return [
            (x_training_data, y_training_data),
            (x_validation_data, y_validation_data),
            (x_test_data, y_test_data)
        ]
