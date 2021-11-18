import numpy as np


def data_spliter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of shape m * n.
        y: has to be an numpy.array, a vector of shape m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible shapes.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    try:
        if len(x.shape) == 1:
            x = x.reshape(-1,1)
        y = y.reshape(-1,1)
        data = np.array(np.append(x, y, axis=1))
        rng = np.random.default_rng()
        rng.shuffle(data)
        p = int(np.round(proportion * x.shape[0]))
        data_train, data_test = data[:p], data[p:]
        return (data_train[:,:-1], data_test[:,:-1], data_train[:,-1].reshape(-1, 1), data_test[:,-1].reshape(-1, 1))
    except Exception as err:
        print("Error: add_polynomial_features: {0}: {1}".format(type(err).__name__, err))
        return None


if __name__=="__main__":

    x1 = np.array([1, 42, 300, 10, 59])
    y = np.array([0,1,0,1,0])
    print(data_spliter(x1, y, 0.8))