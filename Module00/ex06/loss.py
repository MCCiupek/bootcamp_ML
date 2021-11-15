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


def loss_elem_(y, y_hat):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between y and y_hat.
        None if y or y_hat is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    return (y_hat - y) ** 2


def loss_(y, y_hat):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a shape matching problem between y or y_hat.
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
    return float(np.sum(loss_elem_(y, y_hat), axis=0) / (2 * y.shape[0]))


if __name__=="__main__":

    import numpy as np
    x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
    theta1 = np.array([[2.], [4.]])
    y_hat1 = predict_(x1, theta1)
    y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

    print(loss_elem_(y1, y_hat1))
    print(loss_(y1, y_hat1))
    
    x2 = np.array([[0.2, 2., 20.], [0.4, 4., 40.], [0.6, 6., 60.], [0.8, 8., 80.]])
    theta2 = np.array([[0.05], [1.], [1.], [1.]])
    y_hat2 = predict_(x2, theta2)
    y2 = np.array([[19.], [42.], [67.], [93.]])

    print(loss_elem_(y2, y_hat2))
    print(loss_(y2, y_hat2))

    x3 = np.array([0, 15, -9, 7, 12, 3, -21])
    theta3 = np.array([[0.], [1.]])
    y_hat3 = predict_(x3, theta3)
    y3 = np.array([2, 14, -13, 5, 12, 4, -19])
    print(y_hat3)
    print(y3)

    print(loss_(y3, y_hat3))
    print(loss_(y3, y3))
