
from sklearn.feature_selection import chi2, SelectKBest, RFE, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr
import numpy as np

class FeatureSelector:
    def __init__(self, X, y):
        """
        Initializes the FeatureSelector class, which handles various feature selection techniques 
        and classification models.

        Parameters:
        X (array-like or DataFrame): The input features for the model.
        y (array-like or Series): The target variable for the model.
        """
        self.X = X
        self.y = y
        self.chi2_selector = SelectKBest(chi2)
        self.lasso_selector = Lasso()
        self.rf_classifier = RandomForestClassifier(random_state=42)
        self.rfe_selector = RFE(estimator=self.rf_classifier)
        self.sfs_selector = SequentialFeatureSelector(estimator=self.rf_classifier)
        
    def select_features_chi2(self, k=10, threshold=None):
        """
        Selects the top k features based on the chi-squared statistical test.

        Parameters:
        k (int): Number of features to select.
        threshold (float): Minimum score threshold for the chi-squared test.

        Returns:
        dict: A dictionary containing:
            - 'features': Names of the selected features.
            - 'scores': Chi-squared scores of the features.
        """
        self.chi2_selector.k = k
        X_selected = self.chi2_selector.fit_transform(self.X, self.y)
        scores = self.chi2_selector.scores_
        if threshold is not None:
            selected_mask = scores >= threshold
        else:
            selected_mask = self.chi2_selector.get_support()
        selected_features = self.X.columns[selected_mask]
        return {'features': selected_features, 'scores': scores[selected_mask]}
    
    def select_features_spearman(self, threshold=0.5):
        """
        Selects features based on their Spearman correlation with the target variable.

        Parameters:
        threshold (float): Threshold for the correlation. Only features with an absolute correlation
                           above this value are selected.

        Returns:
        dict: A dictionary containing:
            - 'selected_features': Names of the selected features.
            - 'correlations': Correlation coefficients of the selected features.
        """
        correlations = np.array([spearmanr(self.X[col], self.y, nan_policy='omit')[0] for col in self.X.columns])
        valid_correlations = ~np.isnan(correlations)
        correlations = correlations[valid_correlations]
        features = self.X.columns[valid_correlations]
        selected_mask = np.abs(correlations) > threshold
        selected_features = features[selected_mask]
        selected_correlations = correlations[selected_mask]
        return {'selected_features': selected_features, 'correlations': selected_correlations}

    def lasso_feature_selection(self, alpha=0.01, threshold=0.01):
        """
        Selects features using Lasso regression.

        Parameters:
        alpha (float): The penalty parameter.
        threshold (float): Minimum threshold for the coefficients.

        Returns:
        dict: A dictionary containing:
            - 'selected_features': Names of the selected features.
            - 'coefficients': Coefficients of the selected features.
        """
        self.lasso_selector.alpha = alpha
        self.lasso_selector.fit(self.X, self.y)
        coefficients = self.lasso_selector.coef_
        selected_mask = np.abs(coefficients) > threshold  
        selected_features = self.X.columns[selected_mask]  
        selected_coefficients = coefficients[selected_mask]  
        return {'selected_features': selected_features, 'coefficients': selected_coefficients}

    def random_forest_feature_importance(self, n_features=10, threshold=None):
        """
        Selects features based on their importance as calculated by a Random Forest classifier.

        Parameters:
        n_features (int): Number of features to select.
        threshold (float): Minimum threshold for the feature importance.

        Returns:
        dict: A dictionary containing:
            - 'selected_features': Names of the selected features.
            - 'importances': Importance scores of the selected features.
        """
        self.rf_classifier.fit(self.X, self.y)
        importances = self.rf_classifier.feature_importances_
        if threshold is not None:
            selected_mask = importances >= threshold
        else:
            indices = np.argsort(importances)[-n_features:]
            selected_mask = np.zeros_like(importances, dtype=bool)
            selected_mask[indices] = True
        selected_features = self.X.columns[selected_mask]
        return {'selected_features': selected_features, 'importances': importances[selected_mask]}

    def sequential_feature_selection(self, n_features_to_select=10, direction='forward'):
        """
        Selects features sequentially using a Random Forest classifier.

        Parameters:
        n_features_to_select (int): Number of features to select.
        direction (str): Direction of the selection ('forward' or 'backward').

        Returns:
        dict: A dictionary containing:
            - 'selected_features': Names of the selected features.
        """
        self.sfs_selector.n_features_to_select = n_features_to_select
        self.sfs_selector.direction = direction
        self.sfs_selector.fit(self.X, self.y)
        selected_features = self.X.columns[self.sfs_selector.get_support()]
        return {'selected_features': selected_features}
