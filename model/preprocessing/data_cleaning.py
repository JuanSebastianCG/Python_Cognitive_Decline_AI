# ./model/preprocessing/data_cleaning.py
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest
from category_encoders import TargetEncoder, OneHotEncoder
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def balance_data(X, y):
    """
    Balancea las clases en el conjunto de datos utilizando la técnica SMOTE.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.

    Retorna:
    X_balanced (pd.DataFrame o np.array): Conjunto de características balanceado.
    y_balanced (pd.Series o np.array): Etiquetas balanceadas.
    """
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced

def standardize(X):
    """
    Estandariza las características numéricas utilizando StandardScaler.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características numéricas.

    Retorna:
    X_scaled (np.array): Conjunto de características estandarizado.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def impute_missing_values(X,ignore_columns):
    """
    Imputa los valores faltantes en el conjunto de datos utilizando KNNImputer.
    this function is used to fill the missing values in the dataset using the KNNImputer method.
    Parameters:
    X (pd.DataFrame or np.array): Dataset with missing values.
    Returns:
    X_imputed (pd.DataFrame or np.array): Dataset with missing values filled.
    """
    X = X.drop(columns=ignore_columns)
    imputer = KNNImputer()
    X_imputed = imputer.fit_transform(X)
    return X_imputed

def outlier_detection(X):
    """
    Detecta valores atípicos en el conjunto de datos utilizando Isolation Forest.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características a evaluar.

    Retorna:
    y_outliers (np.array): Array indicando los valores atípicos (-1 para outliers, 1 para inliers).
    """
    detector = IsolationForest(random_state=42)
    y_outliers = detector.fit_predict(X)
    return y_outliers

def feature_normalization(X, min_max_values):
    """
    Normaliza las características numéricas utilizando Min-Max Scaling.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características numéricas.
    min_max_values (dict): Diccionario con los valores mínimos y máximos para la normalización.

    Retorna:
    X_normalized (pd.DataFrame o np.array): Conjunto de características normalizado.
    """
    X_normalized = (X - min_max_values['min']) / (min_max_values['max'] - min_max_values['min'])
    return X_normalized

def encode_categorical_features(df, target_column, encoding_type='target'):
    """
    Codifica las características categóricas del conjunto de datos utilizando el método especificado.
    
    Parámetros:
    df (pd.DataFrame): DataFrame que contiene las características a codificar.
    target_column (str): Nombre de la columna objetivo para Target Encoding.
    encoding_type (str): Tipo de codificación ('target' para Target Encoding o 'onehot' para One-Hot Encoding).

    Retorna:
    df_encoded (pd.DataFrame): DataFrame con las características categóricas codificadas y numéricas estandarizadas.
    """
    if encoding_type == 'target':
        encoder = TargetEncoder(cols=df.select_dtypes(include=['object']).columns)
    elif encoding_type == 'onehot':
        encoder = OneHotEncoder(cols=df.select_dtypes(include=['object']).columns)
    
    df_encoded = encoder.fit_transform(df, df[target_column])
    
    # Estandarización de características numéricas
    scaler = StandardScaler()
    df_encoded[df_encoded.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df_encoded.select_dtypes(include=['float64', 'int64']).columns)
    
    return df_encoded

def create_normalization_array(df):
    """
    Crea un array con los valores mínimos y máximos de cada característica en el DataFrame.
    
    Parámetros:
    df (pd.DataFrame): DataFrame que contiene las características numéricas.

    Retorna:
    normalization_array (dict): Diccionario con los valores mínimos y máximos de cada característica.
    """
    normalization_array = {
        'min': df.min(axis=0),
        'max': df.max(axis=0)
    }
    return normalization_array

def remove_duplicates(df):
    """
    Elimina los duplicados en el DataFrame.
    
    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos a limpiar.

    Retorna:
    df_deduped (pd.DataFrame): DataFrame sin duplicados.
    """
    df_deduped = df.drop_duplicates()
    return df_deduped

def check_data_consistency(df):
    """
    Verifica la consistencia de los datos en el DataFrame.
    
    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos a verificar.

    Retorna:
    inconsistent_data (dict): Diccionario con estadísticas básicas y valores únicos para cada columna.
    """
    inconsistent_data = {}
    for column in df.columns:
        if df[column].dtype == 'object':
            inconsistent_data[column] = df[column].unique()
        elif df[column].dtype in ['int64', 'float64']:
            inconsistent_data[column] = {
                'min': df[column].min(),
                'max': df[column].max(),
                'mean': df[column].mean()
            }
    return inconsistent_data

def generate_polynomial_features(X, degree=2):
    """
    Genera características polinomiales a partir de las características originales.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características originales.
    degree (int): Grado del polinomio a generar.

    Retorna:
    X_poly (np.array): Conjunto de características polinomiales.
    """
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    return X_poly

def reduce_dimensionality(X, n_components=2):
    """
    Reduce la dimensionalidad del conjunto de datos utilizando PCA.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características a reducir.
    n_components (int): Número de componentes principales a retener.

    Retorna:
    X_reduced (np.array): Conjunto de características con dimensionalidad reducida.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced


