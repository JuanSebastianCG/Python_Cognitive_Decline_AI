import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import re
import csv
import numpy as np


class DataExtractor:
    def __init__(self, base_path="../../data/", columns_to_drop=None):
        """
        Initializes the data extractor with a base path and optional columns to drop.

        Parameters:
        - base_path (str): Base path where the data files are stored.
        - columns_to_drop (list of str, optional): List of column names to drop during data loading.
        """
        self.base_path = base_path
        self.columns_to_drop = columns_to_drop or []



    def save_array_to_txt(self, array, filename):
        """
        Saves a NumPy array (matrix or vector) to a text file.

        Parameters:
        - array (np.ndarray): The matrix or vector to save.
        - filename (str): The name of the file to save the array to.

        Returns:
        - None
        """
        path = self.base_path +"processed/" + filename
        # Ensure the array is a NumPy array.
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Save the array to a text file with proper formatting. sabe with one , delimiter
        np.savetxt(path, array, delimiter=',', fmt='%s', newline='\n')

    def preprocess_csv(self, input_path, output_path, encoding='latin1'):
        """
        Preprocesses a CSV file by reading and correcting lines that contain '//'.

        Parameters:
        - input_path (str): Relative path of the input file.
        - output_path (str): Relative path of the processed output file.
        - encoding (str): Character encoding of the file.
        """
        full_input_path = self.base_path + input_path
        full_output_path = self.base_path + "processed" + output_path
        with open(full_input_path, 'r', encoding=encoding) as infile, \
             open(full_output_path, 'w', encoding=encoding) as outfile:
            buffer = ''
            for i, line in enumerate(infile):
                line = self.remove_non_ascii(line)  # Remove non-ASCII characters.
                if i == 0:
                    # Write the first line removing any '//'
                    outfile.write(line.replace('//', ''))
                else:
                    if '//' in line:
                        # Process and write lines that contain '//'
                        parts = line.split('//')
                        buffer += ' ' + parts[0].strip()
                        outfile.write(buffer + '\n')
                        buffer = parts[1].strip() if len(parts) > 1 else ''
                    else:
                        # Continue accumulating text if there is no '//'
                        buffer += ' ' + line.strip()
            if buffer:
                # Write any remaining text in the buffer
                outfile.write(buffer + '\n')

    def load_data(self, folder_name, file_name, encoding='latin1', delimiter=';'):
        """
        Loads data from a CSV file, attempting to convert columns to numeric and dropping unwanted columns.

        Parameters:
        - folder_name (str): Name of the subdirectory where the file is located.
        - file_name (str): Name of the file to load.
        - encoding (str): Character encoding of the file.
        - delimiter (str): Field delimiter in the CSV file.

        Returns:
        - pd.DataFrame or None: DataFrame with the loaded data or None if an error occurs.
        """
        path = f"{self.base_path}{folder_name}{file_name}"
        try:
            data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=0, quoting=csv.QUOTE_NONE, decimal=',')
            for column in data.columns:
                original_data = data[column].copy()
                data[column] = pd.to_numeric(data[column], errors='coerce')
                # Revert conversion if not all data could be converted to numeric
                if data[column].isna().any():
                    data[column] = original_data
        except pd.errors.ParserError as e:
            print(f"Error reading the file: {e}")
            return None
        if self.columns_to_drop:
            # Drop the specified columns, ignoring errors if they don't exist
            data.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')
        return data

    @staticmethod
    def remove_non_ascii(text):
        """
        Removes non-ASCII characters from a string.

        Parameters:
        - text (str): The text to process.

        Returns:
        - str: The processed text with non-ASCII characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', '', text)

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


