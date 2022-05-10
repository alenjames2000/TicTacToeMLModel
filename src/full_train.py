from common import load_tictac_multi
import numpy
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from joblib import dump

""" CREATES LINEAR REGRESSOR
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        NA
"""
def linear_reg(X, Y):
    weights = numpy.matmul(numpy.linalg.inv(numpy.matmul(numpy.transpose(X), X)), numpy.transpose(X)).dot(Y)
    numpy.savetxt('../regressor_models/weights.txt', weights)

""" CREATES K_NEAREST NEIGHBORS
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
    NA
"""
def knn(X, Y):
    clf = KNeighborsRegressor()
    clf.fit(X, Y)
    dump(clf, '../regressor_models/knn.joblib') 

""" CREATES SUPPORT MULTI_LAYER PERCEPTRON
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        NA
"""
def mlp(X, Y):
    clf = MLPRegressor().fit(X, Y)
    dump(clf, '../regressor_models/mlp.joblib') 

if __name__ == '__main__':
    X, Y = load_tictac_multi()
    linear_reg(X,Y)
    knn(X,Y)
    mlp(X,Y)
