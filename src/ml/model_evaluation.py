
import warnings
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report = classification_report(y_test, y_pred, zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    print(report)
    print("Confusion matrix:")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
