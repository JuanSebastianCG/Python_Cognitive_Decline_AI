import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.decomposition import PCA

class DataCleaner:
    def __init__(self, X, y,delet_columns = []):   
        """
        Inicializa la clase DataCleaner, que se encarga de realizar diversas tareas de 
        limpieza y preprocesamiento de datos, incluyendo la imputación de valores faltantes, 
        balanceo de clases, eliminación de outliers, y codificación de características.

        Parámetros:
        X (DataFrame): Las características de entrada.
        y (Series or array-like): La variable objetivo.
        delet_columns (list): Lista de columnas a eliminar del DataFrame X.
        """
        X.drop(delet_columns, axis=1, inplace=True)

        self.X = X
        self.y = y
        self.imputer = KNNImputer()
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.detector = IsolationForest(random_state=42)
        self.pca = PCA()
        self.min_max_scaler = MinMaxScaler()
        self.poly = PolynomialFeatures(degree=2)
        self.encoders = {
            'target': TargetEncoder(),
            'onehot': OneHotEncoder()
        }

        
    def delete_duplicate_rows(self):
        """
        Elimina filas duplicadas en el DataFrame X, asegurando que la variable objetivo y
        las características queden sincronizadas.

        Returns:
        tuple: El DataFrame de características X sin duplicados y la variable objetivo y correspondiente.
        """
        self.X.drop_duplicates(inplace=True)
        self.y = self.y[self.X.index]
        return self.X, self.y
    
    def delete_null_data(self):
        """
        Elimina filas con valores nulos en el DataFrame X, sincronizando también la variable objetivo y.

        Returns:
        tuple: El DataFrame de características X sin valores nulos y la variable objetivo y correspondiente.
        """
        self.X.dropna(inplace=True)
        self.y = self.y[self.X.index]
        return self.X, self.y
    
    def delete_Object_columns(self):
        """
        Elimina columnas de tipo objeto del DataFrame X, dejando solo columnas numéricas o de otro tipo.

        Returns:
        DataFrame: El DataFrame X sin columnas de tipo objeto.
        """
        self.X = self.X.select_dtypes(exclude=['object'])
        return self.X
        
    def balance_data(self):
        """
        Balancea el conjunto de datos usando la técnica de sobremuestreo SMOTE para manejar el desbalanceo de clases.
        Smote basicamente toma los datos de la clase minoritaria y genera datos sintéticos para igualar la cantidad de datos de la clase mayoritaria.
        los datos sinteticos se generan con la técnica de interpolación de K-vecinos más cercanos.

        Returns:
        tuple: El DataFrame de características balanceadas X y la variable objetivo balanceada y.
        """
        X_balanced, y_balanced = self.smote.fit_resample(self.X, self.y)
        self.X = X_balanced
        self.y = y_balanced
        return X_balanced, y_balanced

    def standardize(self, ignore_columns=[]):
        """
        Estandariza las características numéricas del DataFrame X, ignorando aquellas que se especifiquen.
        formula de estandarización: (X - mean(X)) / std(X)

        Parámetros:
        ignore_columns (list): Lista de columnas a excluir de la estandarización.

        Returns:
        DataFrame: El DataFrame X con las columnas numéricas estandarizadas.
        """
        columns_to_standardize = self.X.select_dtypes(include=[np.number]).columns.difference(ignore_columns)
        self.X.loc[:, columns_to_standardize] = self.scaler.fit_transform(self.X[columns_to_standardize])
        return self.X

    def impute_missing_values(self):
        """
        Imputa valores faltantes en el DataFrame X utilizando la técnica de K-Nearest Neighbors (KNN).
        los valores faltantes se reemplazan por la media de los k vecinos más cercanos.

        Returns:
        DataFrame: El DataFrame X con los valores faltantes imputados.
        """
        self.X[:] = self.imputer.fit_transform(self.X)
        return self.X

    def delete_outliers(self, delete_indices=True):
        """
        Detecta y opcionalmente elimina outliers en el DataFrame X utilizando Isolation Forest.

        Parámetros:
        delete_indices (bool): Si es True, elimina las filas que son outliers. Si es False, 
                               solo devuelve un DataFrame con la marca de outliers.

        Returns:
        DataFrame: Un DataFrame con un indicador de si una fila es un outlier o no.
        """
        indices = self.detector.fit_predict(self.X)
        if delete_indices:
            self.X = self.X[indices == 1]
            self.y = self.y[indices == 1]
        return pd.DataFrame(indices, columns=['outlier'])

    def normalize_features(self, ignore_columns=[]):
        """
        Normaliza las características numéricas del DataFrame X utilizando Min-Max Scaling, 
        ignorando aquellas que se especifiquen.

        Parámetros:
        ignore_columns (list): Lista de columnas a excluir de la normalización.

        Returns:
        DataFrame: El DataFrame X con las columnas numéricas normalizadas.
        """
        columns_to_normalize = self.X.select_dtypes(include=[np.number]).columns.difference(ignore_columns)
        self.X.loc[:, columns_to_normalize] = self.min_max_scaler.fit_transform(self.X[columns_to_normalize])
        return self.X
    
    def encode_features(self, target_column, method='target'):
        """
        Codifica las características categóricas del DataFrame X utilizando la técnica 
        especificada ('target' o 'onehot').

        Parámetros:
        target_column (str): El nombre de la columna objetivo para Target Encoding.
        method (str): Método de codificación ('target' para Target Encoding, 'onehot' para One-Hot Encoding).

        Returns:
        DataFrame: El DataFrame X con las características categóricas codificadas.
        """
        encoder = self.encoders[method]
        self.X = encoder.fit_transform(self.X, self.y if method == 'target' else None)
        return self.X

    def generate_polynomial_columns(self, replace_Db=False):
        """
        Genera nuevas características polinómicas basadas en las existentes en el DataFrame X.

        Parámetros:
        replace_Db (bool): Si es True, reemplaza las características originales por las nuevas características polinómicas.

        Returns:
        DataFrame: Un nuevo DataFrame con las características polinómicas generadas.
        """
        new_features = self.poly.fit_transform(self.X)
        new_features = pd.DataFrame(new_features, columns=self.poly.get_feature_names_out(self.X.columns))
        if replace_Db:
            self.X = new_features
        return new_features

    def reduce_dimensionality(self, n_components=2, replace=False):
        """
        Reduce la dimensionalidad de las características en el DataFrame X utilizando PCA (Análisis de Componentes Principales).

        Parámetros:
        n_components (int): El número de componentes principales a retener.
        replace (bool): Si es True, reemplaza las características originales por los componentes principales.

        Returns:
        array: Un array con los componentes principales obtenidos.
        """
        self.pca.n_components = n_components
        principal_components = self.pca.fit_transform(self.X)    
        if replace:
            columns = [f'PC{i+1}' for i in range(n_components)]
            self.X = pd.DataFrame(principal_components, columns=columns, index=self.X.index)
        return principal_components
