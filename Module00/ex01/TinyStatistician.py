import numpy as np
import sys
import math

class TinyStatistician():
    @staticmethod
    def mean(x):
        if x is None:
            return None
        try:
            mean = 0
            for elem in x:
                mean += elem
            return mean / len(x)
        except Exception as err:
            print("Error: {0}: {1}".format(type(err).__name__, err))
            return None
    
    @staticmethod
    def median(x):
        return TinyStatistician.percentile(x, 50)

    @staticmethod
    def quartile(x):
        return [TinyStatistician.percentile(x, 25), TinyStatistician.percentile(x, 75)]
        
    @staticmethod
    def percentile(x, p):
        if x is None:
            return None
        try:
            y = np.sort(x)
            n = len(y)
            if n == 0:
                return None
            return y[int(math.ceil((n * p) / 100)) - 1]
        except Exception as err:
            print("Error: {0}: {1}".format(type(err).__name__, err))
            return None

    @staticmethod
    def var(x):
        if x is None or len(x) == 0:
            return None
        try:
            mean, var = TinyStatistician.mean(x), 0
            for elem in x:
                var = var + (elem - mean) ** 2
            return var / len(x)
        except Exception as err:
            print("Error: {0}: {1}".format(type(err).__name__, err))
            return None

    @staticmethod
    def std(x):
        if x is None or len(x) == 0:
            return None
        try:
            return TinyStatistician.var(x) ** .5
        except Exception as err:
            print("Error: {0}: {1}".format(type(err).__name__, err))
            return None

if __name__=="__main__":

    ts = TinyStatistician()

    lst1 = ["a", "b", "c"]
    lst2 = [1.0, 2.0, 3.0]
    lst3 = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    lst4 = [102, 56, 34, 99, 89, 101, 10, 54]
    lst5 = [102, 56, 34, 99, 89, 101, 10]
    lst6 = [1, 42, 300, 10, 59]

    L = [lst1, lst2, lst3, lst4, lst5, lst6]

    for lst in L:
        x = np.array(lst)
        print("mean({0}): {1}".format(lst, ts.mean(lst)))
        print("mean({0}): {1}".format(x, ts.mean(x)))
        print("median({0}): {1}".format(lst, ts.median(lst)))
        print("median({0}): {1}".format(x, ts.median(x)))
        print("quartile({0}): {1}".format(lst, ts.quartile(lst)))
        print("quartile({0}): {1}".format(x, ts.quartile(x)))
        print("percentile({0}): {1}".format(lst, ts.percentile(lst, 10)))
        print("percentile({0}): {1}".format(x, ts.percentile(x, 10)))
        print("percentile({0}): {1}".format(lst, ts.percentile(lst, 28)))
        print("percentile({0}): {1}".format(x, ts.percentile(x, 28)))
        print("percentile({0}): {1}".format(lst, ts.percentile(lst, 83)))
        print("percentile({0}): {1}".format(x, ts.percentile(x, 83)))
        print("var({0}): {1}".format(lst, ts.var(lst)))
        print("var({0}): {1}".format(x, ts.var(x)))
        print("std({0}): {1}".format(lst, ts.std(lst)))
        print("std({0}): {1}".format(x, ts.std(x)))
        print()