import numpy as np
import time
import json

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
from bayes_opt import BayesianOptimization

from model.utils.data_extraction import DataExtractor
from model.utils.tensorboard import get_tensorboard_writer

class HyperparameterOptimizer:
    def __init__(self, model, X_val, y_val, writer, cv_splits=3):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.writer = writer
        self.cv = StratifiedKFold(n_splits=cv_splits)

    def objective(self, n_estimators, max_depth, learning_rate):
        params = {
            'random_forest__n_estimators': int(n_estimators),
            'random_forest__max_depth': int(max_depth),
            'xgboost__n_estimators': int(n_estimators),
            'xgboost__learning_rate': learning_rate,
            'xgboost__max_depth': int(max_depth)
        }
        self.model.set_params(**params)
        score = cross_val_score(self.model, self.X_val, self.y_val, cv=self.cv, scoring='roc_auc_ovr').mean()
        self.writer.add_scalar('Objective_Function', score)
        return score

    def optimize(self, init_points=2, n_iter=5):
        optimizer = BayesianOptimization(
            f=self.objective,
            pbounds={
                'n_estimators': (200, 1000),
                'max_depth': (10, 50),
                'learning_rate': (0.001, 0.2)
            },
            random_state=42
        )
        optimizer.maximize(init_points=init_points, n_iter=n_iter)
        best_params = optimizer.max['params']
        self.writer.add_text('Optimized_Hyperparameters', json.dumps(best_params, indent=4))
        # Properly format parameters for the stacking model
        formatted_params = {
            'random_forest__n_estimators': int(best_params['n_estimators']),
            'random_forest__max_depth': int(best_params['max_depth']),
            'xgboost__n_estimators': int(best_params['n_estimators']),
            'xgboost__learning_rate': best_params['learning_rate'],
            'xgboost__max_depth': int(best_params['max_depth'])
        }
        return formatted_params


class ModelTrainer:
    def __init__(self, data_folder_file="DB sin RM (filtrado) - normalizado 01.pkl", labels_name="Deterioro Cognitivo", log_path='../logs/fit/', n_splits=5):
        pathData = '../../data/'
        self.data = DataExtractor.load_data_pickle(pathData, data_folder_file)
        
        separated_data = DataExtractor.extract_test_validation_training_data(self.data, labels_name)
        self.X_train, self.y_train = separated_data['training']
        self.X_test, self.y_test = separated_data['test']
        self.X_validation, self.y_validation = separated_data['validation']
        
        self.writer = get_tensorboard_writer(log_path)
        self.n_splits = n_splits
        self.model = None

    def build_model(self, base_models=None, final_estimator=None, cv_splits=None):
        if base_models is None:
            base_models = [
                ('random_forest', RandomForestClassifier(random_state=42)),
                ('xgboost', XGBClassifier(eval_metric='logloss', random_state=42))
            ]
        if final_estimator is None:
            final_estimator = LogisticRegression()
        if cv_splits is None:
            cv_splits = self.n_splits

        self.model = StackingClassifier(
            estimators=base_models,
            final_estimator=final_estimator,
            cv=StratifiedKFold(n_splits=cv_splits)
        )
        self.writer.add_text('Model_Structure', str(self.model))

    def train_model(self):
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        duration = time.time() - start_time
        train_accuracy = accuracy_score(self.y_train, self.model.predict(self.X_train))
        self.writer.add_scalar('Training_Time', duration)
        self.writer.add_scalar('Training_Accuracy', train_accuracy)
        
        validation_accuracy = self.model.score(self.X_validation, self.y_validation)
        test_accuracy = self.model.score(self.X_test, self.y_test)
        self.writer.add_scalar('Validation_Accuracy', validation_accuracy)
        self.writer.add_scalar('Test_Accuracy', test_accuracy)

    def validate_model(self):
        accuracy_scores = []
        for i in range(100):
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
        optimizer = HyperparameterOptimizer(self.model, self.X_validation, self.y_validation, self.writer)
        best_params = optimizer.optimize()
        self.model.set_params(**best_params)
        self.train_model()
        self.validate_model()
        self.writer.close()
        
        return self.model

