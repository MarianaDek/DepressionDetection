# src/model_evaluation.py

import warnings
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import UndefinedMetricWarning

def evaluate_model(model, X_test, y_test):
    """
    Друкує classification_report і confusion_matrix, ігноруючи
    UndefinedMetricWarning та підставляючи zero_division=0.
    """
    y_pred = model.predict(X_test)


    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
        report = classification_report(
            y_test,
            y_pred,
            zero_division=0
        )

    cm = confusion_matrix(y_test, y_pred)

    print(report)
    print("Confusion matrix:")
    print(cm)
