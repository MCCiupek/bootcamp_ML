import numpy as np


def pos_neg(y, y_hat):
    try:
        tn = y[np.where((y_hat==0) & (y==0))].size
        tp = y[np.where((y_hat==1) & (y==1))].size
        fn = y[np.where((y_hat==0) & (y==1))].size
        fp = y[np.where((y_hat==1) & (y==0))].size
        return tp, tn, fn, fp
    except Exception as err:
        print("Error: pos_neg: {0}: {1}".format(type(err).__name__, err))
        return None

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    Return:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, tn, fn, fp = pos_neg(y, y_hat)
        return (tp + tn) / (tp + fp + tn + fn)
    except Exception as err:
        print("Error: accuracy_score_: {0}: {1}".format(type(err).__name__, err))
        return None

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.array for the correct labels
    y_hat:a numpy.array for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    try:
        tp, tn, fn, fp = pos_neg(y, y_hat)
        return tp / (tp + fp)
    except Exception as err:
        print("Error: precision_score_: {0}: {1}".format(type(err).__name__, err))
        return None

def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        tp, tn, fn, fp = pos_neg(y, y_hat)
        return tp / (tp + fn)
    except Exception as err:
        print("Error: recall_score_: {0}: {1}".format(type(err).__name__, err))
        return None

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The f1 score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return (2 * precision_score_(y, y_hat) * recall_score_(y, y_hat)) / (precision_score_(y, y_hat) + recall_score_(y, y_hat))
    except Exception as err:
        print("Error: f1_score_: {0}: {1}".format(type(err).__name__, err))
        return None