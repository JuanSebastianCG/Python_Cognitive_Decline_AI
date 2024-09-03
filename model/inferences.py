import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, f1_score
from sklearn.preprocessing import label_binarize
from itertools import cycle


class ModelMetrics:
    def __init__(self, model, X_test, y_test, n_classes):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.n_classes = n_classes
        self.y_pred = model.predict(X_test)
        self.y_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None

    def print_classification_report(self):
        print("Classification Report:\n", classification_report(self.y_test, self.y_pred))

    def plot_roc_curve(self):
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
        f1 = f1_score(self.y_test, self.y_pred, average='macro')  # Usando 'macro' para considerar todas las clases
        print("F1 Score: {f1:.2f}")