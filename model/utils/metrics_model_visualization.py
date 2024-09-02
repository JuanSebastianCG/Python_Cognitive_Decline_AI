import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, f1_score

class ModelMetrics:
    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)[:, 1]  # Asumiendo que es un problema de clasificación binaria

    def print_classification_report(self):
        report = classification_report(self.y_test, self.y_pred)
        print("Classification Report:\n", report)

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_test, self.y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()

    def plot_precision_recall_curve(self):
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_proba)
        plt.figure()
        plt.plot(recall, precision, label='Precision-Recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.show()

    def print_f1_score(self):
        f1 = f1_score(self.y_test, self.y_pred)
        print(f"F1 Score: {f1:.2f}")
