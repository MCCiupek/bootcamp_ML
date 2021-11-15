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
    try:
        return np.append(np.ones((len(x), 1)), x, axis=1)
    except Exception as err:
        return None

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of shape m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta shapes are not appropriate.
        None if x or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if x is None or theta is None:
        return None
    if not isinstance(x, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(theta) == 0:
        return None
    try:
        X = add_intercept(x.reshape((len(x), 1)))
        return np.matmul(X, theta)
    except Exception as err:
        return None

if __name__=="__main__":

    x = np.arange(1,6)
    theta1 = np.array([5, 0])
    print(simple_predict(x, theta1))

    theta2 = np.array([0, 1])
    print(simple_predict(x, theta2))

    theta3 = np.array([5, 3])
    print(simple_predict(x, theta3))

    theta4 = np.array([-3, 1])
    print(simple_predict(x, theta4))