import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from itertools import cycle

class ModelMetrics:
    def __init__(self, model, X_test, y_test, n_classes):
        """
        Initializes the ModelMetrics class to handle performance metrics visualization and reporting for a classification model.
        
        Parameters:
        model (classifier): The classification model being evaluated.
        X_test (array-like): Test dataset features.
        y_test (array-like): True labels for the test dataset.
        n_classes (int): Number of classes in the target classification.
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    def print_classification_report(self):
        """
        Prints the classification report including precision, recall, and F1-score for each class.
        """
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

    def plot_roc_curve(self):
        """
        Plots the Receiver Operating Characteristic (ROC) curve for each class in a multi-class setting.
        """
        y_test_bin = label_binarize(self.y_test, classes=[*range(self.n_classes)])
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'yellow', 'orange'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Multi-class')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        """
        Plots the Precision-Recall curve for all classes in a multi-class setting, using micro-averaging.
        """
        y_test_bin = label_binarize(self.y_test, classes=[*range(self.n_classes)])
        precision, recall, _ = precision_recall_curve(y_test_bin.ravel(), self.y_proba.ravel())
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='gold', lw=2,
                 label='Micro-average Precision-recall curve (area = {0:0.2f})'.format(auc(recall, precision)))
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve for Multi-class')
        plt.legend(loc="best")
        plt.show()

    def print_f1_score(self):
        """
        Prints the F1 score, using macro-average to consider the unweighted mean of each class score.
        """
        f1 = f1_score(self.y_test, self.y_pred, average='macro')
        print(f"F1 Score: {f1:.2f}")

    def plot_confusion_matrix(self):
        """
        Plots a confusion matrix to visualize the accuracy of the model classification.
        """
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.show()

    def plot_learning_curve(self):
        """
        Plots the learning curve of the model to evaluate how well the model learns as more data is available.
        """
        train_sizes, train_scores, test_scores = learning_curve(self.model, self.X_test, self.y_test, cv=5, n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 10))
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.figure(figsize=(10, 8))
        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training score')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Cross-validation score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc="best")
        plt.show()

    def plot_feature_importance(self):
        """
        Displays a bar chart of the feature importances if the model supports it.
        """
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.figure(figsize=(10, 8))
            plt.title('Feature Importances')
            plt.bar(range(self.X_test.shape[1]), importances[indices], color='b', align='center')
            plt.xticks(range(self.X_test.shape[1]), self.X_test.columns[indices], rotation=90)
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()
        else:
            print("The model does not support feature importance.")

