import numpy as np


def mse_(y, y_hat):
    """
    Description:
        Calculate the MSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        mse: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        return None
    return float(np.sum((y_hat - y) ** 2) / y.shape[0])


def rmse_(y, y_hat):
    """
    Description:
        Calculate the RMSE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        rmse: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """
    return float(mse_(y, y_hat) ** .5)


def mae_(y, y_hat):
    """
    Description:
        Calculate the MAE between the predicted output and the real output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        mae: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        return None
    return float(np.sum(np.abs(y_hat - y)) / y.shape[0])



def r2score_(y, y_hat):
    """
    Description:
        Calculate the R2score between the predicted output and the output.
    Args:
        y: has to be a numpy.array, a vector of shape m * 1.
        y_hat: has to be a numpy.array, a vector of shape m * 1.
    Returns:
        r2score: has to be a float.
        None if there is a matching shape problem.
    Raises:
        This function should not raise any Exception.
    """
    if y.shape != y_hat.shape:
        return None
    return float(1 - mse_(y, y_hat) * y.shape[0] / np.sum((y - np.mean(y)) ** 2))



if __name__=="__main__":

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from math import sqrt

    x = np.array([0, 15, -9, 7, 12, 3, -21])
    y = np.array([2, 14, -13, 5, 12, 4, -19])

    # Mean squared error
    print(mse_(x,y))
    print(mean_squared_error(x,y))

    # Root mean squared error
    print(rmse_(x,y))
    print(sqrt(mean_squared_error(x,y)))
   
    # Mean absolute error
    print(mae_(x,y))
    print(mean_absolute_error(x,y))
    
    # R2-score
    print(r2score_(x,y))
    print(r2_score(x,y))
