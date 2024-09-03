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
        """
        Save a given array to a text file in CSV format.

        Parameters:
        base_path (str): The base directory path where the file will be saved.
        array (array-like): The array to be saved. It can be a list or a numpy array.
        filename (str): The name of the file to save the array to.

        Returns:
        None
        """
        # Construct the full path for the output file
        path = base_path + "processed/" + filename
        
        # Ensure the input is a numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Save the array to a text file with comma as the delimiter
        np.savetxt(path, array, delimiter=',', fmt='%s', newline='\n')

    @staticmethod
    def preprocess_csv(base_path, input_path, output_path, encoding='latin1'):
        """
        Preprocess a CSV file by removing non-ASCII characters and handling lines with '//' comments.

        Parameters:
        base_path (str): The base directory path where the input and output files are located.
        input_path (str): The relative path to the input CSV file.
        output_path (str): The relative path to the output CSV file.
        encoding (str, optional): The encoding of the input and output files. Default is 'latin1'.

        Returns:
        None
        """
        # Construct the full paths for the input and output files
        full_input_path = base_path + input_path
        full_output_path = base_path + "processed" + output_path
        
        # Open the input file for reading and the output file for writing
        with open(full_input_path, 'r', encoding=encoding) as infile, \
            open(full_output_path, 'w', encoding=encoding) as outfile:
            
            buffer = ''  # Initialize a buffer to accumulate lines
            
            # Iterate over each line in the input file
            for i, line in enumerate(infile):
                # Remove non-ASCII characters from the line
                line = DataExtractor.remove_non_ascii(line)
                
                if i == 0:
                    # For the first line, remove '//' and write to the output file
                    outfile.write(line.replace('//', ''))
                else:
                    if '//' in line:
                        # If '//' is found in the line, split the line at '//' and process
                        parts = line.split('//')
                        buffer += ' ' + parts[0].strip()
                        outfile.write(buffer + '\n')
                        buffer = parts[1].strip() if len(parts) > 1 else ''
                    else:
                        # If '//' is not found, accumulate the line in the buffer
                        buffer += ' ' + line.strip()
            
            # Write any remaining buffer content to the output file
            if buffer:
                outfile.write(buffer + '\n')

    @staticmethod
    def load_data_csv(base_path, folder_name, file_name, encoding='latin1', delimiter=';'):
        """
        Load data from a CSV file and attempt to convert columns to numeric types.

        Parameters:
        base_path (str): The base directory path where the folder is located.
        folder_name (str): The name of the folder containing the CSV file.
        file_name (str): The name of the CSV file to be loaded.
        encoding (str, optional): The encoding of the CSV file. Default is 'latin1'.
        delimiter (str, optional): The delimiter used in the CSV file. Default is ';'.

        Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame. If there is a parsing error, returns None.
        """
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
        """
        Load data from a text file and return it as a NumPy array of strings.

        Parameters:
        base_path (str): The base directory path where the folder is located.
        folder_name (str): The name of the folder containing the file.
        file_name (str): The name of the text file to be loaded.
        delimiter (str, optional): The delimiter used to separate values in the text file. Default is '\n'.
        encoding (str, optional): The encoding of the text file. Default is 'utf-8'.

        Returns:
        np.ndarray: A NumPy array containing the data from the text file.
        """
        path = f"{base_path}{folder_name}{file_name}"
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, encoding=encoding)
    
    @staticmethod
    def load_data_txt(base_path, folder_name, file_name, delimiter='\n', encoding='utf-8'):
        """
        Load data from a text file and return it as a NumPy array of strings.

        Parameters:
        base_path (str): The base directory path where the folder is located.
        folder_name (str): The name of the folder containing the file.
        file_name (str): The name of the text file to be loaded.
        delimiter (str, optional): The delimiter used to separate values in the text file. Default is '\n'.
        encoding (str, optional): The encoding of the text file. Default is 'utf-8'.

        Returns:
        np.ndarray: A NumPy array containing the data from the text file.
        """
        path = f"{base_path}{folder_name}{file_name}"
        return np.genfromtxt(path, delimiter=delimiter, dtype=str, encoding=encoding)

    @staticmethod
    def save_data_pickle(base_path, data, namefile, filter_condition=None):
        """
        Save data to a pickle file, optionally filtering it based on a condition.

        Parameters:
        base_path (str): The base directory path where the file will be saved.
        data (DataFrame): The data to be saved.
        namefile (str): The name of the file to save the data in.
        filter_condition (str, optional): A condition to filter the data before saving. Defaults to None.

        Returns:
        None
        """
        # Construct the full path for the output file
        path = f"{base_path}processed/{namefile}"
        
        # If a filter condition is provided, filter the data
        if filter_condition:
            data = data.query(filter_condition)
        
        # Open the file in write-binary mode and save the data using pickle
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_data_pickle(base_path, namefile):
        with open(base_path + "processed/" + namefile, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def save_data_json(base_path, data, namefile):
        """
        Save data to a JSON file.

        Parameters:
        base_path (str): The base directory path where the file will be saved.
        data (pandas.DataFrame): The data to be saved.
        namefile (str): The name of the file to save the data in.

        Returns:
        None
        """
        path = f"{base_path}processed/{namefile}"
        with open(path, 'w') as f:
            data.to_json(f, orient='records', lines=True)

    @staticmethod
    def load_data_json(base_path, namefile):
        """
        Load data from a JSON file.

        Parameters:
        base_path (str): The base directory path where the file is located.
        namefile (str): The name of the JSON file to be loaded.

        Returns:
        DataFrame: A pandas DataFrame containing the data from the JSON file.
        """
        path = f"{base_path}processed/{namefile}"
        return pd.read_json(path, lines=True)

    @staticmethod
    def extract_test_validation_training_data(data, target, training_ratio=0.7, test_ratio=0.2, validation_ratio=0.1):
        """
        Split data into training, validation, and test sets.

        Parameters:
        data (pd.DataFrame): The input data as a pandas DataFrame.
        target (str): The name of the target column.
        training_ratio (float, optional): The proportion of the data to include in the training set. Default is 0.7.
        test_ratio (float, optional): The proportion of the data to include in the test set. Default is 0.2.
        validation_ratio (float, optional): The proportion of the data to include in the validation set. Default is 0.1.

        Returns:
        dict: A dictionary containing the training, validation, and test sets. Each set is represented as a list with two elements:
            - The first element is a DataFrame of the features (X).
            - The second element is a Series of the target variable (y).
            The dictionary has the following structure:
            {
                'training': [x_training_data, y_training_data],
                'validation': [x_validation_data, y_validation_data],
                'test': [x_test_data, y_test_data]
            }
        """
        # Split the data into training and temporary sets
        training_data, temp_data = train_test_split(data, test_size=1-training_ratio, random_state=42)
        relative_validation_ratio = validation_ratio / (test_ratio + validation_ratio)
        validation_data, test_data = train_test_split(temp_data, test_size=relative_validation_ratio, random_state=42)
        
        # Separate features and target for the training set
        x_training_data = training_data.drop(columns=[target])
        y_training_data = training_data[target]
        
        # Separate features and target for the validation set
        x_validation_data = validation_data.drop(columns=[target])
        y_validation_data = validation_data[target]
        
        # Separate features and target for the test set
        x_test_data = test_data.drop(columns=[target])
        y_test_data = test_data[target]
        
        # Return the split data as a dictionary
        return {
            'training': [x_training_data, y_training_data],
            'validation': [x_validation_data, y_validation_data],
            'test': [x_test_data, y_test_data]
        }

    @staticmethod
    def save_model(base_path, model, base_filename):
        """
        Save a machine learning model to a file with a timestamped filename.

        Parameters:
        base_path (str): The base directory where the model file will be saved.
        model (object): The machine learning model to be saved.
        base_filename (str): The base name for the model file.

        Returns:
        str: The name of the saved model file.
        """
        # Get the current date and time as a string formatted as 'YYYY-MM-DD_HH-MM-SS'
        date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Create the full filename by appending the date string and '.pkl' extension to the base filename
        namefile = f"{base_filename}_{date_str}.pkl"
        
        # Create the full path by combining the base path and the filename
        path = f"{base_path}{namefile}"
        
        # Save the model to the specified path using joblib
        joblib.dump(model, path)
        
        # Return the name of the saved model file
        return namefile
    
    @staticmethod
    def load_model(base_path, fileName):
        """
        Load a machine learning model from a specified file.

        Parameters:
        base_path (str): The directory path where the model file is located.
        fileName (str): The name of the model file to be loaded.

        Returns:
        object: The loaded machine learning model.
        """
        path = f"{base_path}/{fileName}"
        return joblib.load(path)
    
    @staticmethod
    def remove_non_ascii(text):
        """
        Remove non-ASCII characters from a given text string.

        Parameters:
        text (str): The input string from which non-ASCII characters need to be removed.

        Returns:
        str: A new string with all non-ASCII characters removed.
        """
        return re.sub(r'[^\x00-\x7F]+', '', text)