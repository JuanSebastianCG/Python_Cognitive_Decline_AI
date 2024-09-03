import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import re
import csv
import numpy as np
import joblib
import datetime


class DataExtractor:
    @staticmethod
    def save_array_to_txt(base_path, array, filename):
        path = base_path + "processed/" + filename
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        np.savetxt(path, array, delimiter=',', fmt='%s', newline='\n')

    @staticmethod
    def preprocess_csv(base_path, input_path, output_path, encoding='latin1'):
        full_input_path = base_path + input_path
        full_output_path = base_path + "processed" + output_path
        with open(full_input_path, 'r', encoding=encoding) as infile, \
             open(full_output_path, 'w', encoding=encoding) as outfile:
            buffer = ''
            for i, line in enumerate(infile):
                line = DataExtractor.remove_non_ascii(line)
                if i == 0:
                    outfile.write(line.replace('//', ''))
                else:
                    if '//' in line:
                        parts = line.split('//')
                        buffer += ' ' + parts[0].strip()
                        outfile.write(buffer + '\n')
                        buffer = parts[1].strip() if len(parts) > 1 else ''
                    else:
                        buffer += ' ' + line.strip()
            if buffer:
                outfile.write(buffer + '\n')

    @staticmethod
    def load_data_csv(base_path, folder_name, file_name, encoding='latin1', delimiter=';'):
        path = f"{base_path}{folder_name}{file_name}"
        try:
            data = pd.read_csv(path, encoding=encoding, delimiter=delimiter, header=0, quoting=csv.QUOTE_NONE, decimal=',')
            for column in data.columns:
                original_data = data[column].copy()
                data[column] = pd.to_numeric(data[column], errors='coerce')
                if data[column].isna().any():
                    data[column] = original_data
        except pd.errors.ParserError as e:
            print(f"Error reading the file: {e}")
            return None
        return data

    @staticmethod
    def load_data_txt(base_path, folder_name, file_name, delimiter='\n', encoding='utf-8'):
        path = f"{base_path}{folder_name}{file_name}"
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, encoding=encoding)
    
    @staticmethod
    def remove_non_ascii(text):
        return re.sub(r'[^\x00-\x7F]+', '', text)

    @staticmethod
    def save_data_pickle(base_path, data, namefile, filter_condition=None):
        path = f"{base_path}processed/{namefile}"
        if filter_condition:
            data = data.query(filter_condition)
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data_pickle(base_path, namefile):
        with open(base_path + "processed/" + namefile, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_data_json(base_path, data, namefile):
        path = f"{base_path}processed/{namefile}"
        with open(path , 'w') as f:
            data.to_json(f, orient='records', lines=True)

    @staticmethod
    def load_data_json(base_path, namefile):
        path = f"{base_path}processed/{namefile}"
        return pd.read_json(path, lines=True)

    @staticmethod
    def extract_test_validation_training_data(data, target, training_ratio=0.7, test_ratio=0.2, validation_ratio=0.1):
        training_data, temp_data = train_test_split(data, test_size=1-training_ratio, random_state=42)
        relative_validation_ratio = validation_ratio / (test_ratio + validation_ratio)
        validation_data, test_data = train_test_split(temp_data, test_size=relative_validation_ratio, random_state=42)
        
        x_training_data = training_data.drop(columns=[target])
        y_training_data = training_data[target]
        x_validation_data = validation_data.drop(columns=[target])
        y_validation_data = validation_data[target]
        x_test_data = test_data.drop(columns=[target])
        y_test_data = test_data[target]
        
        return {
            'training': [x_training_data, y_training_data],
            'validation': [x_validation_data, y_validation_data],
            'test': [x_test_data, y_test_data]
        }

    @staticmethod
    def save_model(base_path, model, directory, base_filename):
        directory = base_path + directory
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{directory}{base_filename}_{date_str}.pkl"
        joblib.dump(model, filename)        
        return filename
    
    @staticmethod
    def load_model(base_path, directory, fileName):
        path = f"{base_path}trained_models/{directory}/{fileName}"
        return joblib.load(path)
