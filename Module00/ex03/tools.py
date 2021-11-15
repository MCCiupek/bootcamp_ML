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


if __name__=="__main__":

    x = np.arange(1,6).reshape((5,1))
    print(add_intercept(x))
    print(x)

    y = np.arange(1,10).reshape((3,3))
    print(add_intercept(y))
    print(y)
