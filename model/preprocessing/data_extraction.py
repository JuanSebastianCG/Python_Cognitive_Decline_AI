import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import unidecode
from io import StringIO
import csv

class DataExtractor:
    def __init__(self, base_path="../../data/", columns_to_drop=None):
        """
        Inicializa la clase DataExtractor, que se encarga de cargar, procesar y dividir conjuntos de datos
        en conjuntos de entrenamiento, validación y prueba.

        Parámetros:
        base_path (str): Ruta base donde se almacenan los datos.
        columns_to_drop (list): Lista de columnas a eliminar al cargar los datos.
        """
        self.base_path = base_path
        self.columns_to_drop = columns_to_drop if columns_to_drop is not None else []
        self.removed_features = []  # Almacena los nombres de las características eliminadas
        self.removed_rows = []  # Almacena las filas problemáticas

    def load_data(self, nameFolderRaw, file_name, encoding='latin1', delimiter=';', expected_fields=3):
        """
        Carga datos desde un archivo CSV, eliminando columnas innecesarias y aquellas problemáticas.

        Parámetros:
        nameFolderRaw (str): Nombre de la carpeta que contiene los datos en bruto.
        file_name (str): Nombre del archivo CSV a cargar.
        encoding (str): Codificación del archivo. Por defecto es 'latin1'.
        delimiter (str): Delimitador utilizado en el archivo CSV. Por defecto es ';'.
        expected_fields (int): Número esperado de campos por línea.

        Returns:
        DataFrame: El DataFrame con los datos cargados y las columnas especificadas eliminadas.
        """
        folder_path = self.base_path + nameFolderRaw + file_name
        valid_rows = []
        
        with open(folder_path, 'r', encoding=encoding) as file:
            for i, line in enumerate(file, 1):
                # Contar el número de campos en la línea
                if len(line.split(delimiter)) == expected_fields:
                    valid_rows.append(line)
                else:
                    self.removed_rows.append((i, line.strip()))

        # Cargar los datos filtrados en un DataFrame usando StringIO
        try:
            data = pd.read_csv(StringIO('\n'.join(valid_rows)), 
                               encoding=encoding, 
                               delimiter=delimiter, 
                               quoting=csv.QUOTE_NONE)
        except pd.errors.ParserError as e:
            print(f"Error al leer el archivo filtrado: {e}")
            return None

        # Eliminar columnas problemáticas (ejemplo: columnas con muchos valores nulos)
        """ columns_to_remove = data.columns[data.isnull().mean() > 0.5].tolist()
        self.removed_features.extend(columns_to_remove)
        data = data.drop(columns=columns_to_remove) """


        # Reemplazar 'ñ' con 'n' y eliminar tildes
        data.columns = [unidecode.unidecode(col).replace('n', 'ñ') for col in data.columns]
        data = data.applymap(lambda x: unidecode.unidecode(x).replace('n', 'ñ') if isinstance(x, str) else x)

        return data

    def get_removed_features(self):
        """
        Devuelve una lista de las características que fueron eliminadas debido a problemas.
        
        Returns:
        list: Lista de características eliminadas.
        """
        return self.removed_features
    
    def get_removed_rows(self):
        """
        Devuelve una lista de las filas que fueron eliminadas debido a problemas.
        
        Returns:
        list: Lista de filas eliminadas con su número de línea y contenido.
        """
        return self.removed_rows

    @staticmethod
    def save_data_pickle(data, namefile, base_path, filter_condition=None):
        """
        Guarda un conjunto de datos en un archivo pickle, con una opción para aplicar un filtro antes de guardar.

        Parámetros:
        data (DataFrame or object): El conjunto de datos o el objeto a guardar.
        namefile (str): Nombre del archivo pickle donde se guardará.
        base_path (str): Ruta base donde se almacenará el archivo.
        filter_condition (str, optional): Condición de filtrado a aplicar antes de guardar los datos.
        """
        if filter_condition:
            data = data.query(filter_condition)
        with open(base_path + "processed/" + namefile, 'wb') as f:
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


