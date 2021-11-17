import numpy as np


def loss_(y, y_hat):
    """
    Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    y = y.flatten()
    y_hat = y_hat.flatten()
    if y.shape != y_hat.shape:
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    return float(np.dot(y_hat - y, y_hat - y) / (2 * y.shape[0]))