from sklearn.feature_selection import chi2, SelectKBest, RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
import numpy as np

class FeatureSelector:
    def __init__(self, X, y):
        """
        Inicializa la clase FeatureSelector, que se encarga de manejar diversas técnicas 
        de selección de características y modelos de clasificación.

        Parámetros:
        X (array-like or DataFrame): Las características de entrada para el modelo.
        y (array-like or Series): La variable objetivo para el modelo.
        """
        self.X = X
        self.y = y
        self.chi2_selector = SelectKBest(chi2)
        self.lasso_selector = Lasso()
        self.rf_classifier = RandomForestClassifier(random_state=42)
        self.rfe_selector = RFE(estimator=self.rf_classifier)
        self.sfs_selector = SequentialFeatureSelector(estimator=self.rf_classifier)
        
    def select_features_chi2(self, k=10):
        """
        Selecciona las k mejores características según la prueba estadística chi-cuadrado.
        Esta prueba mide la dependencia entre variables estocásticas, útil para seleccionar
        características que están más fuertemente relacionadas con la variable objetivo.

        Parámetros:
        k (int): Número de características a seleccionar. El valor por defecto es 10.

        Returns:
        dict: Un diccionario que contiene:
            - 'features': Los nombres de las características seleccionadas.
            - 'scores': Las puntuaciones chi-cuadrado de las características.
        """
        self.chi2_selector.k = k
        X_selected = self.chi2_selector.fit_transform(self.X, self.y)
        scores = self.chi2_selector.scores_
        selected_features = self.X.columns[self.chi2_selector.get_support()]
        return {'features': selected_features, 'scores': scores}
    
    
    def select_features_spearman(self, threshold=0.5):
        """
        Selecciona las características basadas en su correlación de Spearman con la variable objetivo.
        Spearman evalúa la relación monótona entre las características y la variable objetivo.
        Esto lo hace haciendo uso de los rangos de las observaciones en lugar de los valores brutos.

        Parámetros:
        threshold (float): Umbral para la correlación. Solo se seleccionan características 
                           con una correlación absoluta superior a este valor.

        Returns:
        dict: Un diccionario que contiene:
            - 'selected_features': Los nombres de las características seleccionadas.
            - 'correlations': Los coeficientes de correlación de las características seleccionadas.
        """
        correlations = np.array([spearmanr(self.X[col], self.y, nan_policy='omit')[0] for col in self.X.columns])
        valid_correlations = ~np.isnan(correlations)
        correlations = correlations[valid_correlations]
        features = self.X.columns[valid_correlations]
        selected_mask = np.abs(correlations) > threshold
        selected_features = features[selected_mask]
        selected_correlations = correlations[selected_mask]
        return {'selected_features': selected_features, 'correlations': selected_correlations}

    def lasso_feature_selection(self, alpha=0.01):
        """
        Selecciona características utilizando la regresión Lasso, que penaliza la suma de 
        los valores absolutos de los coeficientes. Lasso puede llevar a coeficientes exactamente 
        iguales a cero, lo que resulta en la selección automática de características.

        Parámetros:
        alpha (float): El parámetro de penalización. Un alpha mayor resulta en más coeficientes 
                       siendo reducidos a cero.

        Returns:
        dict: Un diccionario que contiene:
            - 'selected_features': Los nombres de las características seleccionadas.
            - 'coefficients': Los coeficientes de las características seleccionadas.
        """
        self.lasso_selector.alpha = alpha
        self.lasso_selector.fit(self.X, self.y)
        coefficients = self.lasso_selector.coef_
        selected_mask = np.abs(coefficients) > 0  
        selected_features = self.X.columns[selected_mask]  
        selected_coefficients = coefficients[selected_mask]  
        return {'selected_features': selected_features, 'coefficients': selected_coefficients}

    def sequential_feature_selection(self, n_features_to_select=10, direction='forward'):
        """
        Selecciona características secuencialmente utilizando un clasificador de Random Forest.
        Esta técnica puede realizar una selección 'forward' o 'backward', donde en cada paso
        se agregan o eliminan características para mejorar el rendimiento del modelo.

        Parámetros:
        n_features_to_select (int): Número de características a seleccionar.
        direction (str): Dirección de la selección. Puede ser 'forward' para añadir características
                         o 'backward' para eliminarlas.

        Returns:
        dict: Un diccionario que contiene:
            - 'selected_features': Los nombres de las características seleccionadas.
        """
        self.sfs_selector.n_features_to_select = n_features_to_select
        self.sfs_selector.direction = direction
        self.sfs_selector.fit(self.X, self.y)
        selected_features = self.X.columns[self.sfs_selector.get_support()]
        return {'selected_features': selected_features}

    def random_forest_feature_importance(self, n_features=10):
        """
        Selecciona características basadas en su importancia calculada por un clasificador 
        de Random Forest. Las características se seleccionan según sus puntuaciones de 
        importancia.

        Parámetros:
        n_features (int): Número de características a seleccionar.

        Returns:
        dict: Un diccionario que contiene:
            - 'selected_features': Los nombres de las características seleccionadas.
            - 'importances': Las importancias de las características seleccionadas.
        """
        self.rf_classifier.fit(self.X, self.y)
        importances = self.rf_classifier.feature_importances_
        indices = np.argsort(importances)[-n_features:]
        selected_features = self.X.columns[indices]
        return {'selected_features': selected_features, 'importances': importances[indices]}

    
import matplotlib.pyplot as plt
import numpy as np

class FeatureVisualizer:
    
    @staticmethod
    def plot_scores(features, scores, title='Feature Scores'):
        """
        Dibuja un gráfico de barras de las puntuaciones de las características.

        Parámetros:
        features (array-like): Los nombres de las características.
        scores (array-like): Las puntuaciones correspondientes a las características.
        title (str): Título del gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(features, scores, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Scores')
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def plot_coefficients(features, coefficients, title='Lasso Coefficients'):
        """
        Dibuja un gráfico de barras de los coeficientes de Lasso.

        Parámetros:
        features (array-like): Los nombres de las características.
        coefficients (array-like): Los coeficientes correspondientes a las características.
        title (str): Título del gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(features, coefficients, color='green')
        plt.xlabel('Features')
        plt.ylabel('Coefficient Value')
        plt.title(title)
        plt.xticks(rotation=90)
        plt.show()

    @staticmethod
    def plot_importances(features, importances, title='Random Forest Feature Importances'):
        """
        Dibuja un gráfico de barras horizontales de la importancia de las características
        calculadas por Random Forest.

        Parámetros:
        features (array-like): Los nombres de las características.
        importances (array-like): Las importancias correspondientes a las características.
        title (str): Título del gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.barh(features, importances, color='red')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title(title)
        plt.show()

    @staticmethod
    def plot_correlations(features, correlations, title='Spearman Correlations'):
        """
        Dibuja un gráfico de barras de las correlaciones de Spearman.

        Parámetros:
        features (array-like): Los nombres de las características.
        correlations (array-like): Los coeficientes de correlación correspondientes.
        title (str): Título del gráfico.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(features, correlations, color='orange')
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficient')
        plt.title(title)
        plt.xticks(rotation=90)
        plt.axhline(0, color='grey', linewidth=0.8)  # Línea de referencia en cero
        plt.show()
