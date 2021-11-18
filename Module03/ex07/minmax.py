import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def norm(x, min_, max_):
    """Computes the normalized version of a non-empty numpy.array using the min-max standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    return x * (max_ - min_) + min_


if __name__=="__main__":

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(minmax(X))

    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(minmax(Y))