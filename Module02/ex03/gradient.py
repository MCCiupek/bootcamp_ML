import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
    Returns:
        x as a numpy.array, a vector of shape m * (n + 1).
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return np.append(np.ones((len(x), 1)), x, axis=1)
    except Exception as err:
        return None


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a (n +1) * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape n * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        X = add_intercept(x)
        return 1 / y.shape[0] * np.matmul(np.transpose(X), (np.matmul(X, theta) - y))
    except Exception as err:
        print("Error: {0}: {1}".format(type(err).__name__, err))
        return None