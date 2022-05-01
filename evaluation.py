"""
evaluation.py
    Home for testing and accuracy metrics

    @author Nicholas Nordstrom
"""
from sklearn.metrics import ConfusionMatrixDisplay


def confusion_matrix(y_pred, y_true, display=False):
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    if display:
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()

    return cm
