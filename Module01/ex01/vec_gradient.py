import numpy as np


def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
    Returns:
        x as a numpy.array, a vector of shape m * 2.
        None if x is not a numpy.array.
        None if x is a empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if x is None:
        return None
    if not isinstance(x, np.ndarray):
        return None
    if x.size == 0 or len(x) == 0:
        return None
    if len(x.shape) == 1:
        x = x.reshape((len(x), 1))
    try:
        return np.append(np.ones((len(x), 1)), x, axis=1)
    except Exception as err:
        return None


def gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
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


if __name__=="__main__":

    x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
    y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
    theta1 = np.array([2, 0.7])
    print(gradient(x, y, theta1))

    theta2 = np.array([1, -0.4])
    print(gradient(x, y, theta2))
