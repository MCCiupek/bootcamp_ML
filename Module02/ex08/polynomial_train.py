from mylinearregression import MyLinearRegression as MyLR
from polynomial_model import add_polynomial_features
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot(x, cx, y, y_hat, labels, colors):
    try:
        plt.plot(x, y, 'o', color=colors[0], label=labels[0])
        plt.plot(cx, y_hat, '-', color=colors[1], label=labels[1])
        plt.xlabel(labels[2])
        plt.ylabel(labels[3])
        plt.legend()
        plt.show()
    except Exception as err:
        print(err)


def regress(x, y, thetas, alpha, max_iter, labels, colors):
    try:
        print("------------")
        lm = MyLR(thetas, alpha=alpha, max_iter=max_iter)
        lm.fit_(x, y)
        y_pred = lm.predict_(x)
        mse = lm.loss_(y, y_pred)
        print("Score: {0}".format(mse))
        continuous_x = add_polynomial_features(np.arange(np.min(x[:,0]), np.max(x[:,0]), 0.01).reshape(-1,1), x.shape[1])
        y_pred = lm.predict_(continuous_x)
        plot(x[:,0], continuous_x[:,0], y, y_pred, labels, colors)
        print("------------")
    except Exception as err:
        print(err)


data = pd.read_csv("../ressources/are_blue_pills_magics.csv")

x = np.array(data['Micrograms']).reshape(-1, 1)
y = np.array(data['Score']).reshape(-1, 1)

lst_x = [add_polynomial_features(x, 1)]

for i in range(2, 7):
    lst_x.append(add_polynomial_features(x, i))

alpha=0.0000000001
max_iter=10000

for i in range(0, 6):
    regress(lst_x[i], y, MyLR.normal_eq(lst_x[i], y), alpha, max_iter,
        ['Score', 'Predicted score', "x: quantity of blue pills patient has taken (in micrograms)", "y: score at the spacecraft driving test"],
        ['b', 'c'])