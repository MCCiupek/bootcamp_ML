import numpy as np


def vec_log_loss_(y, y_hat, eps=1e-15):
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
        if len(y.shape) > 1:
            y = y.flatten()
        if len(y_hat.shape) > 1:
            y_hat = y_hat.flatten()
        return - 1 / y.shape[0] * (np.dot(y, np.log(y_hat + eps)) + np.dot(np.ones(y.shape) - y, np.log(np.ones(y.shape) - y_hat)))
    except Exception as err:
        print("Error: vec_log_loss_: {0}: {1}".format(type(err).__name__, err))
        return None