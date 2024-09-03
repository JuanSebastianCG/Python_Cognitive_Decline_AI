import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, f1_score

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

class ModelMetrics:
    def __init__(self, model, X_test, y_test, n_classes):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test)

    def print_classification_report(self):
        report = classification_report(self.y_test, self.y_pred)
        print("Classification Report:\n", report)


    def plot_roc_curve(self):
        # Binarizar las etiquetas en un formato one-vs-all
        y_test_bin = label_binarize(self.y_test, classes=[*range(self.n_classes)])
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # Calcular la ROC y el área bajo la curva (AUC) para cada clase
        for i in range(self.n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], self.y_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Gráfico ROC para cada clase
        plt.figure(figsize=(10, 8))
        colors = cycle(['blue', 'red', 'green', 'yellow', 'orange'])
        for i, color in zip(range(self.n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for multi-class')
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
