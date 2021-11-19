import numpy as np


class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression class to fit like a boss.
    """
    def __init__(self, thetas, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.array(thetas)#.reshape((len(thetas), 1))

    @staticmethod
    def add_intercept(x):
        """
        Adds a column of 1’s to the non-empty numpy.array x.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
        Returns:
            x as a numpy.array, a vector of shape m * (n + 1).
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

    @staticmethod
    def normal_eq(x, y, eps=1e-15):
        try:
            X = MyLogisticRegression.add_intercept(x)
            return np.dot(np.linalg.inv(np.dot(np.transpose(X), X)), np.dot(np.transpose(X), -np.log((1 - y + eps) / (y + eps))))
        except Exception as err:
            print("Error: normal_eq: {0}: {1}".format(type(err).__name__, err))
            return None


    def loss_(self, x, y, eps=1e-15):
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
            y_hat = self.predict_(x)
            if len(y_hat.shape) > 1:
                y_hat = y_hat.flatten()
            return - 1 / y.shape[0] * (np.dot(y, np.log(y_hat + eps)) + np.dot(np.ones(y.shape) - y, np.log(np.ones(y.shape) - y_hat + eps)))
        except Exception as err:
            print("Error: loss_: {0}: {1}".format(type(err).__name__, err))
            return None

    @staticmethod
    def predict(x, theta):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
        Returns:
            y_hat as a numpy.array, a vector of shape m * 1.
            None if x or theta are empty numpy.array.
            None if x or theta shapes are not appropriate.
            None if x or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            X = MyLogisticRegression.add_intercept(x)
            y_hat = np.dot(X, theta)
            return y_hat
        except Exception as err:
            print("Error: predict: {0}: {1}".format(type(err).__name__, err))
            return None

    #@staticmethod
    def sigmoid_(self, x):
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
            # if len(x.shape) > 1 and x.shape[1]:
            #     res = np.zeros(x.shape)
            #     for i in range(0, x.shape[1]):
            #         res[:,i] = 1 / (1 + np.exp(-x[:,i]))
            # return res
            return 1 / (1 + np.exp(-x))
        except Exception as err:
            print("Error: sigmoid_: {0}: {1}".format(type(err).__name__, err))
            return None

    def predict_(self, x):
        """Computes the vector of prediction y_hat from two non-empty numpy.array.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            theta: has to be an numpy.array, a vector of shape (n + 1) * 1.
        Return:
            y_hat: a numpy.array of shape m * 1, when x and theta numpy arrays
            with expected and compatible shapes.
            None: otherwise.
        Raises:
            This function should not raise any Exception.
        """
        try:
            y_hat = MyLogisticRegression.predict(x, self.thetas)
            return self.sigmoid_(y_hat)
        except Exception as err:
            print("Error: predict_: {0}: {1}".format(type(err).__name__, err))
            return None


    def log_gradient(self, x, y):
        """Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a vector of shape m * n.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a (n +1) * 1 vector.
        Return:
            The gradient as a numpy.array, a vector of shape n * 1.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if x, y or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            X = MyLogisticRegression.add_intercept(x)
            return (1 / y.size) * np.transpose(X).dot(self.predict_(x) - y)
        except Exception as err:
            print("Error: log_gradient: {0}: {1}".format(type(err).__name__, err))
            return None


    def fit_(self, x, y):
        """
        Description:
            Fits the model to the training dataset contained in x and y.
        Args:
            x: has to be a numpy.array, a vector of shape m * n: (number of training examples, number of features).
            y: has to be a numpy.array, a vector of shape m * 1: (number of training examples, 1).
            theta: has to be a numpy.array, a vector of shape (n + 1) * 1.
            alpha: has to be a float, the learning rate
            max_iter: has to be an int, the number of iterations done during the gradient descent
        Return:
            new_theta: numpy.array, a vector of shape (n + 1) * 1.
            None if there is a matching shape problem.
            None if x, y, theta, alpha or max_iter is not of the expected type.
        Raises:
            This function should not raise any Exception.
        """
        try:
            for i in range(self.max_iter):
                self.thetas = self.thetas - self.alpha * self.log_gradient(x, y)
            return self.thetas
        except Exception as err:
            print("Error: fit_: {0}: {1}".format(type(err).__name__, err))
            return None
