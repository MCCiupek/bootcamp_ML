import numpy as np


def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be an numpy.array, a vector
    Return:
        The sigmoid value as a numpy.array.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return 1 / (1 + np.exp(-x))
    except Exception as err:
        print("Error: sigmoid_: {0}: {1}".format(type(err).__name__, err))
        return None