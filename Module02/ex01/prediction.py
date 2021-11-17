import numpy as np


def add_intercept(x):
    """
    Adds a column of 1’s to the non-empty numpy.array x.
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
        if len(x.shape) == 1:
            x = x.reshape((len(x), 1))
        return np.append(np.ones((len(x), 1)), x, axis=1)
    except Exception as err:
        print("Error: add_intercept: {0}: {1}".format(type(err).__name__, err))
        return None


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
        theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
    Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        X = add_intercept(x)
        y_hat = np.matmul(X, theta)
        # if len(y_hat.shape) == 1:
        #     return y_hat.reshape((x.shape[0], 1))
        return y_hat
    except Exception as err:
        print("Error: simple_predict: {0}: {1}".format(type(err).__name__, err))
        return None
