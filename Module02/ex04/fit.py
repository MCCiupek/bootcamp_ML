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
        return np.matmul(X, theta)
    except Exception as err:
        return None


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a (n + 1) * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape (n + 1) * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        X = add_intercept(x)
        return 1 / y.shape[0] * np.matmul(np.transpose(X), predict_(x, theta) - y)
    except Exception as err:
        print("Error: {0}: {1}".format(type(err).__name__, err))
        return None


def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.array, a vector of shape m * n: (number of training examples, number of features).
        y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
        theta: has to be a numpy.array, a vector of shape (n + 1) * 1: (number of features, 1).
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Return:
        new_theta: numpy.array, a vector of shape (n + 1) * 1.
        None if there is a matching shape problem.
        None if x, y, theta, alpha or max_iter is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        for it in range(0, max_iter):
            theta = theta - alpha * gradient(x, y, theta)
        return theta
    except Exception as err:
        print("Error: {0}: {1}".format(type(err).__name__, err))
        return None