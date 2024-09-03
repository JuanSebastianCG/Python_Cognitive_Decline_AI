import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from model.preprocessing.data_extraction import DataExtractor
from model.utils.tensorboard import get_tensorboard_writer
import json
from bayes_opt import BayesianOptimization

class ModelTrainer:
    def __init__(self, dataFolderFile: str = "preprocessed_data.pkl", labelsName: str = "Deterioro Cognitivo"):
        self.dataExtractor = DataExtractor()
        self.data = self.dataExtractor.load_data_pickle(dataFolderFile)
        separated_data = self.dataExtractor.extract_test_validation_training_data(self.data, labelsName)
        self.X_train, self.y_train = separated_data['training']
        self.X_test, self.y_test = separated_data['test']
        self.X_validation, self.y_validation = separated_data['validation']
        self.writer = get_tensorboard_writer('../logs/fit/')
        self.model = None

    def build_model(self):
        base_models = [
            ('random_forest', RandomForestClassifier(random_state=42)),
            ('xgboost', XGBClassifier(eval_metric='logloss', random_state=42))
        ]
        final_estimator = LogisticRegression()
        self.model = StackingClassifier(
            estimators=base_models, final_estimator=final_estimator, cv=StratifiedKFold(n_splits=5))
        self.writer.add_text('Model_Structure', str(self.model))

    def optimize_hyperparameters(self):
        def objective(n_estimators, max_depth, learning_rate):
            model = StackingClassifier(
                estimators=[
                    ('random_forest', RandomForestClassifier(n_estimators=int(n_estimators), max_depth=int(max_depth), random_state=42)),
                    ('xgboost', XGBClassifier(n_estimators=int(n_estimators), learning_rate=learning_rate, max_depth=int(max_depth), random_state=42))
                ],
                final_estimator=LogisticRegression(),
                cv=StratifiedKFold(n_splits=3)
            )
            score = cross_val_score(model, self.X_validation, self.y_validation, cv=2, scoring='roc_auc_ovr').mean()
            return score

        optimizer = BayesianOptimization(
            f=objective,
            pbounds={
                'n_estimators': (200, 1000),
                'max_depth': (10, 50),
                'learning_rate': (0.001, 0.2)
            },
            random_state=42
        )
        optimizer.maximize(init_points=2, n_iter=10)  # Más puntos iniciales e iteraciones
        best_params = optimizer.max['params']
        self.model.set_params(
            random_forest__n_estimators=int(best_params['n_estimators']),
            random_forest__max_depth=int(best_params['max_depth']),
            xgboost__n_estimators=int(best_params['n_estimators']),
            xgboost__learning_rate=best_params['learning_rate'],
            xgboost__max_depth=int(best_params['max_depth'])
        )
        self.writer.add_text('Optimized_Hyperparameters', json.dumps(best_params, indent=4))

    def train_model(self):
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        duration = time.time() - start_time
        train_accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))
        self.writer.add_scalar('Training_Time', duration)
        self.writer.add_scalar('Training_Accuracy', train_accuracy)
        # Agregar registro de precisión en conjunto de validación y test
        validation_accuracy = self.model.score(self.X_validation, self.y_validation)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        self.writer.add_scalar('Validation_Accuracy', validation_accuracy)
        self.writer.add_scalar('Test_Accuracy', test_accuracy)

    def validate_model(self):
        accuracy_scores = []
        for i in range(30):
            resample_idx = np.random.choice(self.X_validation.index, size=len(self.X_validation), replace=True)
            X_resample = self.X_validation.loc[resample_idx]
            y_resample = self.y_validation.loc[resample_idx]
            score = self.model.score(X_resample, y_resample)
            accuracy_scores.append(score)
            self.writer.add_scalar('Bootstrap_Validation_Accuracy', score, i)
        mean_accuracy = np.mean(accuracy_scores)
        self.writer.add_scalar('Mean_Bootstrap_Validation_Accuracy', mean_accuracy)

    def run(self):
        self.build_model()
        self.optimize_hyperparameters()
        self.train_model()
        self.validate_model()
        self.writer.close()

