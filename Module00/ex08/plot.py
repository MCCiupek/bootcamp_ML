import numpy as np
import matplotlib.pyplot as plt


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


def predict_(x, theta):
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
        X = add_intercept(x)
        return np.matmul(X, theta)
    except Exception as err:
        return None


def loss_(y, y_hat):
    """
    Computes the half mean squared error of two non-empty numpy.array, without any for loop.
    The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
        return None
    y = y.flatten()
    y_hat = y_hat.flatten()
    if y.shape != y_hat.shape:
        return None
    if y.size == 0 or y_hat.size == 0:
        return None
    return float(np.dot(y_hat - y, y_hat - y) / (2 * y.shape[0]))


def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a vector of shape 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exception.
    """
    try:
        plt.plot(x, y, 'o')
        plt.plot(x, theta[0] + x * theta[1])
        y_hat = predict_(x, theta)
        for i in range(0, len(x)):
            plt.vlines(x=x[i], ymin=min(y[i], y_hat[i]), ymax=max(y[i], y_hat[i]), color='red', linestyle='dashed')
        plt.title("Cost: {0}".format(loss_(y, y_hat)))
        plt.show()
    except Exception as err:
        print(err)
        pass

if __name__=="__main__":

    x = np.arange(1,6)
    y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
    theta1= np.array([18,-1])
    plot_with_loss(x, y, theta1)
    
    theta2 = np.array([14, 0])
    plot_with_loss(x, y, theta2)

    theta3 = np.array([12, 0.8])
    plot_with_loss(x, y, theta3)