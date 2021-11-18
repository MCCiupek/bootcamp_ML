import numpy as np


def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.array, a vector of shape m * 1.
        y_hat: has to be an numpy.array, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Return:
        The logistic loss value as a float.
        None otherwise.
    Raises:
        This function should not raise any Exception.
    """
    try:
        return - 1 / y.shape[0] * (np.sum(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))) 
    except Exception as err:
        print("Error: log_loss_: {0}: {1}".format(type(err).__name__, err))
        return None