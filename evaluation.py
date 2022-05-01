"""
evaluation.py
    Home for testing and accuracy metrics

    @author Nicholas Nordstrom
"""
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_score


def confusion_matrix(y_pred, y_true, display=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

    return cm


def test_suit(models, X_train, y_train, X_test, y_test):
    for m in models:
        m.fit(X_train, y_train)
        print("Cross Val Score: ", cross_val_score(m, X_train, y_train, cv=5).mean())
        print("Train Score: ", round(m.score(X_train, y_train), 4))
        print("Train Score: ", round(m.score(X_test, y_test), 4))
