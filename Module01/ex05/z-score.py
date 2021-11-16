import numpy as np


def zscore(x):
    """Computes the normalized version of a non-empty numpy.array using the z-score standardization.
    Args:
    x: has to be an numpy.array, a vector.
    Return:
    x’ as a numpy.array.
    None if x is a non-empty numpy.array or not a numpy.array.
    None if x is not of the expected type.
    Raises:
    This function shouldn’t raise any Exception.
    """
    try:
        return (x - np.mean(x)) / np.std(x)
    except Exception as err:
        print("Error: gradient: {0}: {1}".format(type(err).__name__, err))
        return None


if __name__=="__main__":

    X = np.array([0, 15, -9, 7, 12, 3, -21])
    print(zscore(X))

    Y = np.array([2, 14, -13, 5, 12, 4, -19])
    print(zscore(Y))