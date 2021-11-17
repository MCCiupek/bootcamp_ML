from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot(x, y, y_hat, labels, colors):
    try:
        plt.plot(x, y, 'o', color=colors[0], label=labels[0])
        plt.plot(x, y_hat, '.', color=colors[1], label=labels[1])
        plt.xlabel(labels[2])
        plt.ylabel(labels[3])
        plt.legend()
        plt.show()
    except Exception as err:
        print(err)


def univar(x, y, thetas, alpha, max_iter, labels, colors):
    try:
        print("--- {0} ---".format(labels[2]))
        lm = MyLR(thetas, alpha=alpha, max_iter=max_iter)
        lm.fit_(x, y)
        y_pred = lm.predict_(x)
        mse = lm.loss_(y, y_pred)
        print("Thetas:")
        print(lm.thetas)
        print("Score: {0}".format(mse))
        plot(x, y, y_pred, labels, colors)
    except Exception as err:
        print(err)

data = pd.read_csv("../ressources/spacecraft_data.csv")

x = np.array(data[['Age', 'Thrust_power', 'Terameters']])
y = np.array(data['Sell_price']).reshape(-1, 1)

x_0 = np.array(data['Age']).reshape(-1, 1)
x_1 = np.array(data['Thrust_power']).reshape(-1, 1)
x_2 = np.array(data['Terameters']).reshape(-1, 1)


# --- Part One: Univariate Linear Regression ---

print("--- Univariate Linear Regression ---")

# Hyperparameters

alpha=0.0001
max_iter=1000

# Age

univar(x_0, y, np.array([[700.0], [-1]]), alpha, max_iter,
        ['Sell price', 'Predicted sell price', "$x_1$: age (in years)", "y: sell price (in keuros)"],
        ['b', 'c'])

# Thrust

univar(x_1, y, np.array([[100], [5.0]]), alpha, max_iter,
        ['Sell price', 'Predicted sell price', "$x_2$: thrust power (in 10Km/s)", "y: sell price (in keuros)"],
        ['g', 'limegreen'])

# Total distance

univar(x_2, y, np.array([[500.0], [-10]]), alpha, max_iter,
        ['Sell price', 'Predicted sell price', "$x_3$: distance totalier value of spacecraft (in Tmeters)", "y: sell price (in keuros)"],
        ['purple', 'pink'])


# --- Part Two: Multivariate Linear Regression (A New Hope) ---

print("\n--- Multivariate Linear Regression ---")

# Hyperparameters

alpha = 0.00001
max_iter = 1000
thetas = MyLR.normal_eq(x, y)

# Model

lm = MyLR(thetas, alpha=alpha, max_iter=max_iter)
lm.fit_(x, y)
y_pred = lm.predict_(x)
mse = lm.loss_(y, y_pred)

# Print 

print("Thetas:")
print(lm.thetas)
print("Score: {0}".format(mse))

# Plot

plot(x_0, y, y_pred, ['Sell price', 'Predicted sell price', "$x_1$: age (in years)", "y: sell price (in keuros)"],
        ['b', 'c'])

plot(x_1, y, y_pred, ['Sell price', 'Predicted sell price', "$x_2$: thrust power (in 10Km/s)", "y: sell price (in keuros)"],
        ['g', 'limegreen'])

plot(x_2, y, y_pred, ['Sell price', 'Predicted sell price', "$x_3$: distance totalier value of spacecraft (in Tmeters)", "y: sell price (in keuros)"],
        ['purple', 'pink'])
