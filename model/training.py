import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, f1_score
from sklearn.utils import resample
from xgboost import XGBClassifier
from utils.tensorboard import get_tensorboard_writer
import time

class ModelTrainer:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            data, labels, test_size=0.2, stratify=labels, random_state=42)
        self.writer = get_tensorboard_writer()
        self.model = None

    def build_model(self):
        base_models = [
            ('random_forest', RandomForestClassifier(random_state=42)),
            ('xgboost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))
        ]
        final_estimator = LogisticRegression()
        self.model = StackingClassifier(
            estimators=base_models, final_estimator=final_estimator, cv=StratifiedKFold(n_splits=5))
        self.writer.add_text('Model_Structure', str(self.model))

    def hyperparameter_tuning(self):
        skf = StratifiedKFold(n_splits=5)
        param_grid = {
            'random_forest__n_estimators': [100, 200],
            'xgboost__n_estimators': [100, 150]
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=skf, scoring='accuracy')
        start_time = time.time()
        grid_search.fit(self.data, self.labels)
        duration = time.time() - start_time
        self.model = grid_search.best_estimator_
        self.writer.add_text('Best_Model_Parameters', str(grid_search.best_params_))
        self.writer.add_scalar('Hyperparameter_Tuning_Time', duration)

    def train_model(self):
        start_time = time.time()
        self.model.fit(self.X_train, self.y_train)
        duration = time.time() - start_time
        self.writer.add_scalar('Training_Time', duration)
        training_accuracy = self.model.score(self.X_train, self.y_train)
        self.writer.add_scalar('Training_Accuracy', training_accuracy)

    def validate_model(self):
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(self.model, self.data, self.labels, cv=skf, scoring='accuracy')
        for i, score in enumerate(scores, 1):
            self.writer.add_scalar('Cross-Validation Accuracy Fold', score, i)
        self.writer.add_scalar('Average Cross-Validation Accuracy', np.mean(scores))
        self.writer.add_histogram('CV Scores Distribution', scores)

    def bootstrap_validation(self):
        accuracies = []
        for i in range(100):
            X_resampled, y_resampled = resample(self.X_train, self.y_train)
            self.model.fit(X_resampled, y_resampled)
            y_pred = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            accuracies.append(accuracy)
            self.writer.add_scalar('Bootstrap Accuracy Iteration', accuracy, i)
        self.writer.add_histogram('Bootstrap Accuracies Distribution', accuracies)
        self.writer.add_scalar('Mean Bootstrap Accuracy', np.mean(accuracies))
        self.writer.add_scalar('Std Dev Bootstrap Accuracy', np.std(accuracies))

    def run(self):
        self.build_model()
        self.hyperparameter_tuning()
        self.train_model()
        self.validate_model()
        self.bootstrap_validation()
        self.writer.close()

