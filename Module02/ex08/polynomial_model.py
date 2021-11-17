import numpy as np


def add_polynomial_features(x, power):
    """Add polynomial features to vector x by raising its values up to the power given in argument.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        power: has to be an int, the power up to which the components of vector x are going to be raised.
    Return:
        The matrix of polynomial features as a numpy.array, of shape m * n,
        containing the polynomial feature values for all training examples.
        None if x is an empty numpy.array.
        None if x or power is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return np.vander(x.flatten(), N=power + 1, increasing=True)[:,1:]
    except Exception as err:
        print("Error: add_polynomial_features: {0}: {1}".format(type(err).__name__, err))
        return None
    