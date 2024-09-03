import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.decomposition import PCA


class DataCleaner:
    def __init__(self, X, y, delete_columns=[]):   
        """
        Initializes the DataCleaner class, which is responsible for performing various data cleaning
        and preprocessing tasks including imputation of missing values, class balancing, outlier removal,
        and feature encoding.

        Parameters:
        X (DataFrame): The input features.
        y (Series or array-like): The target variable.
        delete_columns (list): List of columns to be removed from the DataFrame X.
        """
        X.drop(delete_columns, axis=1, inplace=True)

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
        Removes duplicate rows in DataFrame X, ensuring that the target variable and
        the features remain synchronized.

        Returns:
        tuple: The feature DataFrame X without duplicates and the corresponding target variable y.
        """
        self.X.drop_duplicates(inplace=True)
        self.y = self.y[self.X.index]
        return self.X, self.y

    def delete_null_data(self):
        """
        Deletes rows with null values in DataFrame X, also synchronizing the target variable y.

        Returns:
        tuple: The feature DataFrame X without null values and the corresponding target variable y.
        """
        self.X.dropna(inplace=True)
        self.y = self.y[self.X.index]
        return self.X, self.y

    def delete_Object_columns(self):
        """
        Removes object type columns from DataFrame X, leaving only numeric or other types of columns.

        Returns:
        DataFrame: The DataFrame X without object type columns.
        """
        self.X = self.X.select_dtypes(exclude=['object'])
        return self.X

    def balance_data(self):
        """
        Balances the dataset using the SMOTE oversampling technique to handle class imbalance.
        SMOTE essentially takes the minority class data and generates synthetic data to match the quantity
        of the majority class data using the K-nearest neighbors interpolation technique.

        Returns:
        tuple: The balanced feature DataFrame X and the balanced target variable y.
        """
        X_balanced, y_balanced = self.smote.fit_resample(self.X, self.y)
        self.X = X_balanced
        self.y = y_balanced
        return X_balanced, y_balanced

    def standardize(self, ignore_columns=[]):
        """
        Standardizes the numeric features of DataFrame X, excluding specified columns.
        Standardization formula: (X - mean(X)) / std(X)

        Parameters:
        ignore_columns (list): List of columns to exclude from standardization.

        Returns:
        DataFrame: The DataFrame X with standardized numeric columns.
        """
        columns_to_standardize = self.X.select_dtypes(include=[np.number]).columns.difference(ignore_columns)
        self.X.loc[:, columns_to_standardize] = self.scaler.fit_transform(self.X[columns_to_standardize])
        return self.X

    def impute_missing_values(self):
        """
        Imputes missing values in DataFrame X using the K-Nearest Neighbors (KNN) technique.
        Missing values are replaced by the mean of the k-nearest neighbors.

        Returns:
        DataFrame: The DataFrame X with imputed missing values.
        """
        self.X[:] = self.imputer.fit_transform(self.X)
        return self.X

    def delete_outliers(self, delete_indices=True):
        """
        Detects and optionally removes outliers in DataFrame X using Isolation Forest.

        Parameters:
        delete_indices (bool): If True, removes rows that are outliers. If False, only returns a DataFrame
                               indicating which rows are outliers.

        Returns:
        DataFrame: A DataFrame indicating whether a row is an outlier or not.
        """
        indices = self.detector.fit_predict(self.X)
        if delete_indices:
            self.X = self.X[indices == 1]
            self.y = self.y[indices == 1]
        return pd.DataFrame(indices, columns=['outlier'])

    def normalize_features(self, ignore_columns=[]):
        """
        Normalizes the numeric features of DataFrame X using Min-Max Scaling, excluding specified columns.

        Parameters:
        ignore_columns (list): List of columns to exclude from normalization.

        Returns:
        DataFrame: The DataFrame X with normalized numeric columns.
        """
        columns_to_normalize = self.X.select_dtypes(include=[np.number]).columns.difference(ignore_columns)
        self.X.loc[:, columns_to_normalize] = self.min_max_scaler.fit_transform(self.X[columns_to_normalize])
        return self.X

    def encode_features(self, target_column, method='target'):
        """
        Encodes the categorical features of DataFrame X using the specified technique ('target' or 'onehot').

        Parameters:
        target_column (str): The name of the target column for Target Encoding.
        method (str): Encoding method ('target' for Target Encoding, 'onehot' for One-Hot Encoding).

        Returns:
        DataFrame: The DataFrame X with encoded categorical features.
        """
        encoder = self.encoders[method]
        self.X = encoder.fit_transform(self.X, self.y if method == 'target' else None)
        return self.X

    def generate_polynomial_columns(self, replace_Db=False):
        """
        Generates new polynomial features based on the existing ones in DataFrame X.

        Parameters:
        replace_Db (bool): If True, replaces the original features with the newly generated polynomial features.

        Returns:
        DataFrame: A new DataFrame with generated polynomial features.
        """
        new_features = self.poly.fit_transform(self.X)
        new_features = pd.DataFrame(new_features, columns=self.poly.get_feature_names_out(self.X.columns))
        if replace_Db:
            self.X = new_features
        return new_features

    def reduce_dimensionality(self, n_components=2, replace=False):
        """
        Reduces the dimensionality of the features in DataFrame X using PCA (Principal Component Analysis).

        Parameters:
        n_components (int): The number of principal components to retain.
        replace (bool): If True, replaces the original features with the principal components.

        Returns:
        array: An array containing the obtained principal components.
        """
        self.pca.n_components = n_components
        principal_components = self.pca.fit_transform(self.X)    
        if replace:
            columns = [f'PC{i+1}' for i in range(n_components)]
            self.X = pd.DataFrame(principal_components, columns=columns, index=self.X.index)
        return principal_components
