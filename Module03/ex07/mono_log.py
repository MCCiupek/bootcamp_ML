from my_logistic_regression import MyLogisticRegression as MyLR
from data_spliter import data_spliter
from minmax import minmax
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys, getopt
from mpl_toolkits.mplot3d import Axes3D


##############################################################################
#                                      plot                                  #       
##############################################################################

def plot_2d(X, x1, x2, y, y_hat, labels, colors=['b', 'c'], threshold=.5):
    try:
        plt.plot(X[np.where(y==1),x1],
                 X[np.where(y==1),x2],
                 'o', color=colors[0],
                 label="Origin")
        plt.plot(X[np.where(y_hat>threshold),x1],
                 X[np.where(y_hat>threshold),x2],
                 '.', color=colors[1],
                 label="Prediction")
        plt.xlabel(labels[x1])
        plt.ylabel(labels[x2])
        plt.show()
    except Exception as err:
        print(err)

def plot_3d(X, y, y_hat, labels, colors=['b', 'c'], threshold=.5):
    try:
        x1 = X[np.where(y==1),0]
        x2 = X[np.where(y==1),1]
        x3 = X[np.where(y==1),2]

        x1_hat = X[np.where(y_hat>threshold),0]
        x2_hat = X[np.where(y_hat>threshold),1]
        x3_hat = X[np.where(y_hat>threshold),2]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(x1, x2, x3, marker='o', color=colors[0], label='True origin')
        ax.scatter(x1_hat, x2_hat, x3_hat, marker='.', color=colors[1], label='Predicted origin')
        ax.set_xlabel("x: {0}".format(labels[0]))
        ax.set_ylabel("y: {0}".format(labels[1]))
        ax.set_zlabel("z: {0}".format(labels[2]))
        plt.legend()
        plt.show()
    except Exception as err:
        print(err)

##############################################################################
#                                   mono_log                                 #       
##############################################################################

def mono_log(zipcode=0, plot=True, verbose=False, test=True):
    
    areas = ['Flying cities of Venus',
            'United Nations of Earth',
            'Mars Republic',
            'The Asteroids’ Belt colonies']
    
    print("------------------------------------------------------------------")
    print("Runing a logistic model to predict whether a citizen comes from the {0} (zipcode={1}) or not.".format(areas[zipcode], zipcode))

    # Data
    
    x = pd.read_csv("../ressources/solar_system_census.csv")
    if verbose: print(x.head())
    
    y = pd.read_csv("../ressources/solar_system_census_planets.csv")
    if verbose: print(y.head())

    y.loc[y['Origin'] == zipcode, 'Origin'] = -1
    y.loc[y['Origin'] > -1, 'Origin'] = 0
    y.loc[y['Origin'] == -1, 'Origin'] = 1
    if verbose: print(y.head())

    norm_x = x.copy()
    # for col in x.columns[1:]:
    #     norm_x[col] = minmax(x[col])
    if verbose: print(norm_x.head())

    norm_y = y.copy()
    # norm_y['Origin'] = minmax(y['Origin'])
    if verbose: print(norm_y.head())
    
    np_x = np.array(norm_x)
    np_y = np.array(norm_y)

    p = .8
    if not test: p = 1

    x_train, x_test, y_train, y_test = data_spliter(np_x, np_y, p, True)
    y_train = y_train
    y_test = y_test
    idx_train = x_train[:,1]
    idx_test = x_test[:,1]
    x_train = x_train[:,1:]
    x_test = x_test[:,1:]
    if verbose: print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    # Train
    
    print("\nTrain:")

    alpha=0.001
    max_iter=10000
    theta = np.zeros(x_train.shape[1] + 1).reshape(-1, 1)

    lm = MyLR(theta, alpha=alpha, max_iter=max_iter)
    lm.fit_(x_train, y_train)
    y_pred = lm.predict_(x_train)
    mse = lm.loss_(x_train, y_train)
    print("\tTrain MSE: ", mse)

    if plot:
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_train, 
            x1=0, x2=1, 
            y=y_train, y_hat=y_pred, 
            labels=x.columns[1:], colors=['b', 'c'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_train, 
            x1=0, x2=2, 
            y=y_train, y_hat=y_pred, 
            labels=x.columns[1:], colors=['g', 'lightgreen'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_train, 
            x1=1, x2=2, 
            y=y_train, y_hat=y_pred, 
            labels=x.columns[1:], colors=['purple', 'pink'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_3d(X=x_train, 
                y=y_train, y_hat=y_pred, 
                labels=x.columns[1:], colors=['b', 'c'], threshold=.5)

    true_negative = y_train[np.where((y_pred<.5) & (y_train==0))].size
    true_positive = y_train[np.where((y_pred>.5) & (y_train==1))].size
    true_pred_train = (true_positive + true_negative) / y_train.size
    print('\ttrue_positive:', true_positive)
    print('\ttrue_negative:', true_negative)
    print('\ttrue_pred:', true_pred_train)

    if not test:
        return lm.thetas, [mse, None], [y_pred, None], [true_pred_train, None]

    # Test

    print("\nTest:")
    y_pred_test = lm.predict_(x_test)
    mse_test = lm.loss_(x_test, y_test)
    print("\tTest MSE: ", mse_test)

    if plot:
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_test, 
            x1=0, x2=1, 
            y=y_test, y_hat=y_pred_test, 
            labels=x.columns[1:], colors=['b', 'c'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_test, 
            x1=0, x2=2, 
            y=y_test, y_hat=y_pred_test, 
            labels=x.columns[1:], colors=['g', 'lightgreen'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_2d(X=x_test, 
            x1=1, x2=2, 
            y=y_test, y_hat=y_pred_test, 
            labels=x.columns[1:], colors=['purple', 'pink'], threshold=.5)
        plt.title("Citizens coming from the {0} ({1})".format(areas[zipcode], zipcode))
        plot_3d(X=x_test, 
                y=y_test, y_hat=y_pred_test, 
                labels=x.columns[1:], colors=['b', 'c'], threshold=.5)

    true_negative = y_test[np.where((y_pred_test<.5) & (y_test==0))].size
    true_positive = y_test[np.where((y_pred_test>.5) & (y_test==1))].size
    true_pred_test = (true_positive + true_negative) / y_test.size
    print('\ttrue_positive:', true_positive)
    print('\ttrue_negative:', true_negative)
    print('\ttrue_pred:', true_pred_test)

    print("------------------------------------------------------------------\n")

    return lm.thetas, [mse, mse_test], [y_pred, y_pred_test], [true_pred_train, true_pred_test]

##############################################################################
#                                       main                                 #       
##############################################################################

if __name__=="__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hz:p:v:",["zipcode=", "plot=", "verbose="])
    except getopt.GetoptError:
        print('mono_log.py -zipcode=<0, 1, 2, 3>')
        sys.exit(2)

    try:
        zipcode = -1
        plot = True
        verbose = False

        for opt, arg in opts:
            if opt == '-h':
                print('mono_log.py -zipcode=<0, 1, 2, 3>')
                sys.exit()
            elif opt in ("-z", "--zipcode"):
                zipcode = int(arg)
                if zipcode not in range(0, 3):
                    print('Error: Arguments: zipcode must be in 0, 1, 2 or 3')
                    sys.exit()
            elif opt in ("-p", "--plot"):
                plot = bool(arg)
            elif opt in ("-v", "--verbose"):
                verbose = bool(arg)
        if zipcode == -1:
            print('Usage: mono_log.py -zipcode=<0, 1, 2, 3>')
            sys.exit()
        
        print("1. With test phase: ")
        theta, mse, y_hat, true_pred = mono_log(zipcode, plot, verbose)
        print("ZIP: {0}".format(zipcode))
        print("Thetas: {0}".format(theta))
        print("MSE: {0}".format(mse))
        print("True Prediction: {0}".format(true_pred))

        print("\n2. Training only: ")
        theta, mse, y_hat, true_pred = mono_log(zipcode, plot, verbose, False)
        print("ZIP: {0}".format(zipcode))
        print("Thetas: {0}".format(theta))
        print("MSE: {0}".format(mse))
        print("True Prediction: {0}".format(true_pred))
    except Exception as err:
        print(err)
        sys.exit()