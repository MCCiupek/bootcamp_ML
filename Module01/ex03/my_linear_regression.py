import numpy as np


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas).reshape((len(thetas), 1))

    def add_intercept(self, x):
        """
        Adds a column of 1’s to the non-empty numpy.array x.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
        Returns:
            x as a numpy.array, a vector of shape m * 2.
            None if x is not a numpy.array.
            None if x is a empty numpy.array.
        Raises:
            This function should not raise any Exception.
        """
        try:
            if len(x.shape) == 1:
                x = x.reshape((len(x), 1))
            return np.append(np.ones((len(x), 1)), x, axis=1)
        except Exception as err:
            print("Error: add_intercept: {0}: {1}".format(type(err).__name__, err))
            return None

    def loss_elem_(self, y, y_hat):
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
        try:
            return (y_hat - y) ** 2
        except Exception as err:
            print("Error: loss_elem_: {0}: {1}".format(type(err).__name__, err))
            return None

    def loss_(self, y, y_hat):
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
        try:
            if not isinstance(y, np.ndarray) or not isinstance(y_hat, np.ndarray):
                return None
            y = y.flatten()
            y_hat = y_hat.flatten()
            if y.shape != y_hat.shape:
                return None
            return float(np.sum(self.loss_elem_(y, y_hat), axis=0) / (2 * y.shape[0]))
        except Exception as err:
            print("Error: loss_: {0}: {1}".format(type(err).__name__, err))
            return None

    def predict_(self, x):
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
        try:
            X = self.add_intercept(x)
            y_hat = np.matmul(X, self.thetas)
            if len(y_hat.shape) == 1:
                return y_hat.reshape(x.shape)
            return y_hat
        except Exception as err:
            print("Error: predict_: {0}: {1}".format(type(err).__name__, err))
            return None


    def gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a 2 * 1 vector.
        Return:
            The gradient as a numpy.array, a vector of shape 2 * 1.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if x, y or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            X = self.add_intercept(x)
            return 1 / y.shape[0] * np.matmul(np.transpose(X), self.predict_(x) - y)
        except Exception as err:
            print("Error: gradient: {0}: {1}".format(type(err).__name__, err))
            return None


    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
            y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
            theta: has to be a numpy.array, a vector of shape 2 * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of shape 2 * 1.
            None if there is a matching shape problem.
            None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            for it in range(0, self.max_iter):
                self.thetas = self.thetas - self.alpha * self.gradient(x, y)
        except Exception as err:
            print("Error: fit_: {0}: {1}".format(type(err).__name__, err))
            return None


if __name__=="__main__":

    from my_linear_regression import MyLinearRegression as MyLR

    x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
    y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
    lr1 = MyLR([2, 0.7])

    print("lr1.predict_(x):")
    print(lr1.predict_(x))
    
    print()

    print("lr1.loss_elem_(lr1.predict_(x),y):")
    print(lr1.loss_elem_(lr1.predict_(x),y))

    print()

    print("lr1.loss_(lr1.predict_(x),y):")
    print(lr1.loss_(lr1.predict_(x),y))

    print()
    print()

    lr2 = MyLR([1, 1], 5e-8, 1500000)
    lr2.fit_(x, y)

    print("lr2.thetas:")
    print(lr2.thetas)

    print()

    print("lr2.predict_(x):")
    print(lr2.predict_(x))

    print()

    print("MyLR.loss_elem_(lr2.predict_(x),y):")
    print(lr2.loss_elem_(lr2.predict_(x), y))

    print()

    print("MyLR.loss_(lr2.predict_(x),y):")
    print(lr2.loss_(lr2.predict_(x), y))