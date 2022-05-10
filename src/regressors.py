import matplotlib.pyplot as plt
import numpy 
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import random
from sklearn.neural_network import MLPRegressor
from common import *
from sklearn.metrics import ConfusionMatrixDisplay

""" CREATES LINEAR REGRESSOR
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        Y_pred: Linear REGRESSOR Predictions
        scor: Accuracy
"""
def linear_reg(X, Y, X_test, Y_test):
    # GET WEIGHTS
    weights = numpy.matmul(numpy.linalg.inv(numpy.matmul(numpy.transpose(X), X)), numpy.transpose(X)).dot(Y)
    # Round to nearest int
    Y_pred = numpy.matmul(X_test, weights)
    Y_pred[Y_pred < 0] = 0
    Y_pred =  numpy.abs(numpy.rint(Y_pred))
    scor = score(Y_pred, Y_test)
    return Y_pred, scor

""" CREATES K_NEAREST NEIGHBORS
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        clf: KNN REGRESSOR
"""
def knn(X, Y):
    clf = KNeighborsRegressor()
    clf.fit(X, Y)
    return clf

""" CREATES SUPPORT MULTI_LAYER PERCEPTRON
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        clf: MLP REGRESSOR
"""
def mlp(X, Y):
    clf = MLPRegressor().fit(X, Y)
    return clf

""" SCORES LINEAR REGRESSOR
    INPUTS:
        Y_pred: Regressor Predictions
        Y_test: Labels
    RETURNS:
        Accuracy
"""
def score(Y_pred, Y_test):
    return numpy.sum(Y_pred == Y_test) / Y_test.size 

""" CREATES CONFUSION MATRIX ARRAY
    INPUTS:
        matrix: Lisit of CONFUSION MATRICIES
    RETURNS:
        Figure of Matricies
"""
def create_matrix(matrix):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15,10))

    for i, axe in zip(range(9), axes.flatten()):
        ConfusionMatrixDisplay(matrix[i]).plot(ax=axe)
        axe.title.set_text(f"{i}")    
    plt.tight_layout() 
    return plt.gcf()
            
""" 10 FOLD CROSS-VALIDATION
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        lin_ac: List of SVM Accuracies for each fold
        lin_cm/better: List of SVM Confusion Matricies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        knne_cm: List of K-NN Confusion Matricies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
        mlpe_cm: List of MLP Confusion Matricies for each fold
"""
def accuracy_matrix(X, Y):
    #Get Folds
    kf = KFold(n_splits=10, shuffle=True)
    #Set Lists
    lin_ac = []
    lin_cm = []
    knne_ac = []
    knne_cm = []
    mlpe_ac = []
    mlpe_cm = []
    #Train on each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)

        a,b = linear_reg(X_train, Y_train, X_test, Y_test)
        lin_ac.append(b)

        pred = knne.predict(X_test)
        pred[pred < 0] = 0
        knne_ac.append(score(numpy.rint(pred), Y_test))

        pred = mlpe.predict(X_test)
        pred[pred < 0] = 0
        mlpe_ac.append(score(numpy.rint(pred), Y_test))
        
        cm = multilabel_confusion_matrix(Y_test, a)
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        lin_cm.append(cm)
        Y_pred = knne.predict(X_test)
        pred[pred < 0] = 0
        cm = multilabel_confusion_matrix(Y_test, numpy.rint(Y_pred))
        knne_cm.append(cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis])
        Y_pred = mlpe.predict(X_test)
        pred[pred < 0] = 0
        cm = multilabel_confusion_matrix(numpy.argmax(Y_test, axis =1), numpy.argmax(numpy.rint(Y_pred), axis=1))
        mlpe_cm.append(cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis])
    
    return lin_ac, lin_cm, knne_ac, knne_cm, mlpe_ac, mlpe_cm

""" 10 TRIALS FOR TRAINING WITH 1/10 of DATA
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        lin_ac: List of SVM Accuraies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
"""
def accuracy(X, Y):
    #Set lists
    lin_ac = []
    knne_ac = []
    mlpe_ac = []
    # Train on each 10th
    for i in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9)
        Y_train, Y_test = Y_train, Y_test

        b = linear_reg(X_train, Y_train, X_test, Y_test)[1]
        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)

        lin_ac.append(b)
        knne_ac.append(score(numpy.rint(knne.predict(X_test)), Y_test))
        mlpe_ac.append(score(numpy.rint(mlpe.predict(X_test)), Y_test))
        
        #plot_confusion_matrix(svmach, X_test, Y_test, normalize='true')
        #plt.show()
    return lin_ac, knne_ac, mlpe_ac

""" 10 FOLD VALIDATION FOR TRAINING WITH NOISY DATA
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        lin_ac: List of SVM Accuraies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
"""
def accuracy_matrix2(X, Y):
    # Set Lists
    lin_ac = []
    knne_ac = []
    mlpe_ac = []
    #Get folds
    kf = KFold(n_splits=10, shuffle=True)
    # Train on each fold
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        b = linear_reg(X_train, Y_train, X_test, Y_test)[1]
        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)

        lin_ac.append(b)
        knne_ac.append(score(numpy.rint(knne.predict(X_test)), Y_test))
        mlpe_ac.append(score(numpy.rint(mlpe.predict(X_test)), Y_test))

    return lin_ac, knne_ac, mlpe_ac

""" ADDS NOISE TO DATA
    INPUTS:
        Y: Labels
    RETURNS:
        Y: Corrupt Labels
"""
def add_noise(Y):
    # Shuffles labels for random selection of data
    index = random.sample(range(0, len(Y)), int(len(Y) * .2))
    for i in index:
        numpy.random.shuffle(Y[i])
    return Y

if __name__ == '__main__':
    # Prints Accuracy and Confusion Matricies for 10 Fold Validation of Original Multi Data
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_multi()
    lin_ac, lin_cm, knne_ac, knne_cm, mlpe_ac, mlpe_cm = accuracy_matrix(X, Y)
    for i in range(10):
        tosay.append(f'Linear Regression Fold {i+1} Accuracy on Classification Multi Dataset: {lin_ac[i]}\nConfusion Matrix:\n{lin_cm[i]}\n')
        print(f'Linear Regression Fold {i+1} Accuracy on Classification Multi Dataset: {lin_ac[i]}\nConfusion Matrix:\n{lin_cm[i]}')
        create_matrix(list(lin_cm[i])).savefig(fr"../Output/Matricies/Regressor/LINEAR/MULTI/LINEAR_MULTI_REGRESSOR_CM_FOLD_{i+1}")

    tosay.append(f'Linear Regression Fold Average Accuracy on Classification Multi Dataset: {numpy.average(lin_ac)}\n')
    print(f'Linear Regression Fold Average Accuracy on Classification Multi Dataset: {numpy.average(lin_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Fold {i+1} Accuracy on Classification Multi Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}\n')
        print(f'K-NN Fold {i+1} Accuracy on Classification Multi Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}')
        create_matrix(list(knne_cm[i])).savefig(fr"../Output/Matricies/Regressor/KNN/MULTI/KNN_MULTI_REGRESSOR_CM_FOLD_{i+1}")

    tosay.append(f'K-NN Fold Average Accuracy on Classification Multi Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Fold Average Accuracy on Classification Multi Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Fold {i+1} Accuracy on Classification Multi Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}\n')
        print(f'Multilayer Perceptron Fold {i+1} Accuracy on Classification Multi Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}')
        create_matrix(list(mlpe_cm[i])).savefig(fr"../Output/Matricies/Regressor/MLP/MULTI/MLP_MULTI_REGRESSOR_CM_FOLD_{i+1}")

    tosay.append(f'Multilayer Perceptron Fold Average Accuracy on Classification Multi Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Fold Average Accuracy on Classification Multi Dataset: {numpy.average(mlpe_ac)}')

    file_write(r"../Output/Multi_Regressor_Accuracy.txt", tosay)

    # Prints Accuracy and Confusion Matricies for 10 Trials of 1/10 Multi Data    
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_multi()
    lin_ac, knne_ac, mlpe_ac = accuracy(X, Y)
    for i in range(10):
        tosay.append(f'Linear Regression Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {lin_ac[i]}\n')
        print(f'Linear Regression Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {lin_ac[i]}')

    tosay.append(f'Linear Regression Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(lin_ac)}\n')
    print(f'Linear Regression Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(lin_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Classification Multi Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on 1/10 Classification Multi Dataset: {numpy.average(mlpe_ac)}') 

    file_write(r"../Output/Multi_Regressor_10TH_Accuracy.txt", tosay)

    # Prints Accuracy and Confusion Matricies for 10 Fold Validation of Courrupt Multi Data
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_multi()
    Y = add_noise(Y)
    lin_ac, knne_ac, mlpe_ac = accuracy(X, Y)
    for i in range(10):
        tosay.append(f'Linear Regression Test {i+1} Accuracy on Noisy Classification Multi Dataset: {lin_ac[i]}\n')
        print(f'Linear Regression Test {i+1} Accuracy on Noisy Classification Multi Dataset: {lin_ac[i]}')

    tosay.append(f'Linear Regression Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(lin_ac)}\n')
    print(f'Linear Regression Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(lin_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on Noisy Classification Multi Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on Noisy Classification Multi Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Classification Multi Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Classification Multi Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on Noisy Classification Multi Dataset: {numpy.average(mlpe_ac)}')
    
    file_write(r"../Output/Multi_Regressor_Noisy_Accuracy.txt", tosay)