# ./model/preprocessing/feature_extraction.py
from sklearn.feature_selection import chi2, SelectKBest, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

def select_features_chi2(X, y, k=10):
    """
    Selecciona las mejores características categóricas utilizando la prueba Chi-Cuadrado.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    k (int): Número de características a seleccionar.

    Retorna:
    X_selected (pd.DataFrame o np.array): Conjunto de características seleccionadas.
    """
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X, y)
    return X_selected

def select_features_spearman(X, y, threshold=0.5):
    """
    Selecciona características basadas en la correlación de Spearman con la variable objetivo.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características numéricas.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    threshold (float): Umbral de correlación para seleccionar características.

    Retorna:
    selected_columns (list): Lista de columnas seleccionadas.
    """
    correlations = [spearmanr(X[col], y)[0] for col in X.columns]
    selected_columns = X.columns[np.abs(correlations) > threshold]
    return X[selected_columns]

def pca_dimensionality_reduction(X, n_components=2):
    """
    Reduce la dimensionalidad utilizando PCA manteniendo el número de componentes especificado.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    n_components (int): Número de componentes principales a retener.

    Retorna:
    X_reduced (pd.DataFrame o np.array): Conjunto de características reducidas.
    """
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced



def lasso_feature_selection(X, y, alpha=0.01):
    """
    Selecciona características utilizando Lasso (L1 Regularization).
    disminuye el peso de las caracteristicas menos importantes hasta hacerlas 0 y tomar las mas importantes
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    alpha (float): Valor de regularización de Lasso.

    Retorna:
    selected_columns (list): Lista de columnas seleccionadas.
    """
    lasso = Lasso(alpha=alpha)
    lasso.fit(X, y)
    selected_columns = X.columns[np.abs(lasso.coef_) > 0]
    return X[selected_columns]

def rfe_feature_selection(X, y, n_features_to_select=10):
    """
    Selecciona características utilizando Eliminación Recursiva de Características (RFE) con RandomForest.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    n_features_to_select (int): Número de características a seleccionar.

    Retorna:
    X_selected (pd.DataFrame o np.array): Conjunto de características seleccionadas.
    """
    model = RandomForestClassifier(random_state=42)
    selector = RFE(model, n_features_to_select=n_features_to_select)
    X_selected = selector.fit_transform(X, y)
    return X_selected

def sequential_feature_selection(X, y, n_features_to_select=10, direction='forward'):
    """
    Selecciona características utilizando el Sequential Feature Selector.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    n_features_to_select (int): Número de características a seleccionar.
    direction (str): Dirección de la selección ('forward' o 'backward').

    Retorna:
    X_selected (pd.DataFrame o np.array): Conjunto de características seleccionadas.
    """
    model = RandomForestClassifier(random_state=42)
    selector = SequentialFeatureSelector(model, n_features_to_select=n_features_to_select, direction=direction)
    X_selected = selector.fit_transform(X, y)
    return X_selected

def random_forest_feature_importance(X, y, n_features=10):
    """
    Selecciona características basadas en la importancia de características de un Random Forest.
    
    Parámetros:
    X (pd.DataFrame o np.array): Conjunto de características.
    y (pd.Series o np.array): Etiquetas correspondientes a las clases.
    n_features (int): Número de características a seleccionar.

    Retorna:
    X_selected (pd.DataFrame o np.array): Conjunto de características seleccionadas.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    selected_columns = X.columns[indices]
    return X[selected_columns]
