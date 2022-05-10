import matplotlib.pyplot as plt
from sklearn import neighbors 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import random
from common import *

""" CREATES SUPPORT VECTOR MACHINE
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        clf: SVM Classifier
"""
def linear_svm(X, Y):
    clf = svm.SVC()
    clf.fit(X, Y)
    return clf

""" CREATES K-NEAREST NEIGHBORS
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        clf: K-NN Classifier
"""
def knn(X, Y):
    clf = KNeighborsClassifier()
    clf.fit(X, Y)
    return clf

""" CREATES MULTI_LAYER PERCEPTRON
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        clf: MLP Classifier
"""
def mlp(X, Y):
    clf = MLPClassifier().fit(X, Y)
    return clf

""" 10 FOLD CROSS-VALIDATION
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        svmach_ac: List of SVM Accuracies for each fold
        svmach_cm/better: List of SVM Confusion Matricies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        knne_cm/better: List of K-NN Confusion Matricies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
        mlpe_cm/better: List of MLP Confusion Matricies for each fold
"""
def accuracy_matrix(X, Y):
    # Create Folds and Lists
    kf = KFold(n_splits=10, shuffle=True)
    
    svmach_ac = []
    svmach_cm = []
    svmach_cm_better = []
    knne_ac = []
    knne_cm = []
    knne_cm_better = []
    mlpe_ac = []
    mlpe_cm = []
    mlpe_cm_better = []

    # Train on each fold 
    for train_index, test_index in kf.split(X):
        # Get Fold
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index].ravel(), Y[test_index].ravel()
        # Train
        svmach = linear_svm(X_train, Y_train)
        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)
        # Append Accuracy and Confusion Matrix
        svmach_ac.append(svmach.score(X_test, Y_test))
        knne_ac.append(knne.score(X_test, Y_test))
        mlpe_ac.append(mlpe.score(X_test, Y_test))
        
        Y_pred = svmach.predict(X_test)
        svmach_cm.append(confusion_matrix(Y_test, Y_pred, normalize='true'))
        Y_pred = knne.predict(X_test)
        knne_cm.append(confusion_matrix(Y_test, Y_pred, normalize='true'))
        Y_pred = mlpe.predict(X_test)
        mlpe_cm.append(confusion_matrix(Y_test, Y_pred, normalize='true'))

        plot_confusion_matrix(svmach, X_test, Y_test, normalize='true')
        svmach_cm_better.append(plt.gcf())
        plt.close()
        plot_confusion_matrix(knne, X_test, Y_test, normalize='true')
        knne_cm_better.append(plt.gcf())
        plt.close()
        plot_confusion_matrix(mlpe, X_test, Y_test, normalize='true')
        mlpe_cm_better.append(plt.gcf())
        plt.close()
        
    return svmach_ac, svmach_cm, svmach_cm_better, knne_ac, knne_cm, knne_cm_better, mlpe_ac, mlpe_cm, mlpe_cm_better

""" 10 TRIALS FOR TRAINING WITH 1/10 of DATA
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        svmach_ac: List of SVM Accuraies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
"""
def accuracy(X, Y):
    # Create Lists
    svmach_ac = []
    knne_ac = []
    mlpe_ac = []

    #Train on 1/10 of Data
    for i in range(10):
        # Get 1/10 of Data and use rest as test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.9)
        Y_train, Y_test = Y_train.ravel(), Y_test.ravel()
        # Train
        svmach = linear_svm(X_train, Y_train)
        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)
        # Append Scores
        svmach_ac.append(svmach.score(X_test, Y_test))
        knne_ac.append(knne.score(X_test, Y_test))
        mlpe_ac.append(mlpe.score(X_test, Y_test))
        
    return svmach_ac, knne_ac, mlpe_ac

""" 10 FOLD VALIDATION FOR TRAINING WITH NOISY DATA
    INPUTS:
        X: Training Feature Vector
        Y: Labels
    RETURNS:
        svmach_ac: List of SVM Accuraies for each fold
        knne_ac: List of K-NN Accuracies for each fold
        mlpe_ac: List of MLP Accuracies for each fold
"""
def accuracy_matrix2(X, Y):
    # Create Lists
    svmach_ac = []
    knne_ac = []
    mlpe_ac = []
    # Get Folds
    kf = KFold(n_splits=10, shuffle=True)
    # Train on each Fold
    for train_index, test_index in kf.split(X):
        # Get Fold
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index].ravel(), Y[test_index].ravel()
        # Train
        svmach = linear_svm(X_train, Y_train)
        knne = knn(X_train, Y_train)
        mlpe = mlp(X_train, Y_train)
        # Append Score
        svmach_ac.append(svmach.score(X_test, Y_test))
        knne_ac.append(knne.score(X_test, Y_test))
        mlpe_ac.append(mlpe.score(X_test, Y_test))

    return svmach_ac, knne_ac, mlpe_ac

""" ADDS NOISE TO DATA
    INPUTS:
        Y: Labels
    RETURNS:
        Y: Corrupt Labels
"""
def add_noise(Y):
    # Chooses 1/5 of the Data to Courrupt
    index = random.sample(range(0, len(Y)), int(len(Y) * .2))
    # Sets the Label to 0
    for i in index:
        Y[i] = numpy.zeros(Y[i].shape)
    return Y

if __name__ == '__main__':
    # Prints Accuracy and Confusion Matricies for 10 Fold Validation of Original Final/Single Data
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_final_single(0)
    svmach_ac, svmach_cm, svmach_cm_better, knne_ac, knne_cm, knne_cm_better, mlpe_ac, mlpe_cm, mlpe_cm_better = accuracy_matrix(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Fold {i+1} Accuracy on Classification Dataset: {svmach_ac[i]}\nConfusion Matrix:\n{svmach_cm[i]}\n')
        print(f'Linear SVM Fold {i+1} Accuracy on Classification Dataset: {svmach_ac[i]}\nConfusion Matrix:\n{svmach_cm[i]}')
        svmach_cm_better[i].savefig(fr"../Output/Matricies/Classifier/SVM/FINAL/SVM_FINAL_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'Linear SVM Fold Average Accuracy on Classification Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Fold Average Accuracy on Classification Dataset: {numpy.average(svmach_ac)}')

    for i in range(10):
        tosay.append(f'K-NN Fold {i+1} Accuracy on Classification Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}\n')
        print(f'K-NN Fold {i+1} Accuracy on Classification Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}')
        knne_cm_better[i].savefig(fr"../Output/Matricies/Classifier/KNN/FINAL/KNN_FINAL_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'K-NN Fold Average Accuracy on Classification Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Fold Average Accuracy on Classification Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Fold {i+1} Accuracy on Classification Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}\n')
        print(f'Multilayer Perceptron Fold {i+1} Accuracy on Classification Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}')
        mlpe_cm_better[i].savefig(fr"../Output/Matricies/Classifier/MLP/FINAL/MLP_FINAL_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'Multilayer Perceptron Fold Average Accuracy on Classification Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Fold Average Accuracy on Classification Dataset: {numpy.average(mlpe_ac)}')

    file_write(r"../Output/Final_Classifier_Accuracy.txt", tosay)

    tosay = []
    X,Y = load_tictac_final_single(1)
    svmach_ac, svmach_cm, svmach_cm_better, knne_ac, knne_cm, knne_cm_better, mlpe_ac, mlpe_cm, mlpe_cm_better = accuracy_matrix(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Fold {i+1} Accuracy on Optimal Single Dataset: {svmach_ac[i]}\nConfusion Matrix:\n{svmach_cm[i]}\n')
        print(f'Linear SVM Fold {i+1} Accuracy on Optimal Single Dataset: {svmach_ac[i]}\nConfusion Matrix:\n{svmach_cm[i]}')
        svmach_cm_better[i].savefig(fr"../Output/Matricies/Classifier/SVM/SINGLE/SVM_SINGLE_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'Linear SVM Fold Average Accuracy on Optimal Single Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Fold Average Accuracy on Optimal Single Dataset: {numpy.average(svmach_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Fold {i+1} Accuracy on Optimal Single Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}\n')
        print(f'K-NN Fold {i+1} Accuracy on Optimal Single Dataset: {knne_ac[i]}\nConfusion Matrix:\n{knne_cm[i]}')
        knne_cm_better[i].savefig(fr"../Output/Matricies/Classifier/KNN/SINGLE/KNN_SINGLE_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'K-NN Fold Average Accuracy on Optimal Single Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Fold Average Accuracy on Optimal Single Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Fold {i+1} Accuracy on Optimal Single Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}\n')
        print(f'Multilayer Perceptron Fold {i+1} Accuracy on Optimal Single Dataset: {mlpe_ac[i]}\nConfusion Matrix:\n{mlpe_cm[i]}')
        mlpe_cm_better[i].savefig(fr"../Output/Matricies/Classifier/MLP/SINGLE/MLP_SINGLE_CLASSIFIER_CM_FOLD_{i+1}")

    tosay.append(f'Multilayer Perceptron Fold Average Accuracy on Optimal Single Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Fold Average Accuracy on Optimal Single Dataset: {numpy.average(mlpe_ac)}\n')

    file_write(r"../Output/Single_Classifier_Accuracy.txt", tosay)

    # Prints Accuracy and Confusion Matricies for 10 Trials of 1/10 Final/Single Data    
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_final_single(0)
    svmach_ac, knne_ac, mlpe_ac = accuracy(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Test {i+1} Accuracy on 1/10 Classification Dataset: {svmach_ac[i]}\n')
        print(f'Linear SVM Test {i+1} Accuracy on 1/10 Classification Dataset: {svmach_ac[i]}')
    
    tosay.append(f'Linear SVM Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(svmach_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on 1/10 Classification Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on 1/10 Classification Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(knne_ac)}')


    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Classification Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Classification Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on 1/10 Classification Dataset: {numpy.average(mlpe_ac)}')
    
    file_write(r"../Output/Final_Classifier_10TH_Accuracy.txt", tosay)

    tosay = []
    X,Y = load_tictac_final_single(1)
    svmach_ac, knne_ac, mlpe_ac = accuracy(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {svmach_ac[i]}\n')
        print(f'Linear SVM Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {svmach_ac[i]}')

    tosay.append(f'Linear SVM Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(svmach_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on 1/10 Optimal Single Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on 1/10 Optimal Single Dataset: {numpy.average(mlpe_ac)}')

    file_write(r"../Output/Single_Classifier_10TH_Accuracy.txt", tosay)

    # Prints Accuracy and Confusion Matricies for 10 Fold Validation of Courrupt Final/Single Data
    # Also Writes Information to a File
    tosay = []
    X,Y = load_tictac_final_single(0)
    Y = add_noise(Y)
    svmach_ac, knne_ac, mlpe_ac = accuracy_matrix2(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Test {i+1} Accuracy on Noisy Classification Dataset: {svmach_ac[i]}\n')
        print(f'Linear SVM Test {i+1} Accuracy on Noisy Classification Dataset: {svmach_ac[i]}')

    tosay.append(f'Linear SVM Test Average Accuracy on Noisy Classification Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Test Average Accuracy on Noisy Classification Dataset: {numpy.average(svmach_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on Noisy Classification Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on Noisy Classification Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on Noisy Classification Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on Noisy Classification Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Classification Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Classification Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on Noisy Classification Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on Noisy Classification Dataset: {numpy.average(mlpe_ac)}')

    file_write(r"../Output/Final_Classifier_Noisy_Accuracy.txt", tosay)
    
    tosay = []
    X,Y = load_tictac_final_single(1)
    Y = add_noise(Y)
    svmach_ac, knne_ac, mlpe_ac = accuracy_matrix2(X, Y)
    for i in range(10):
        tosay.append(f'Linear SVM Test {i+1} Accuracy on Noisy Optimal Single Dataset: {svmach_ac[i]}\n')
        print(f'Linear SVM Test {i+1} Accuracy on Noisy Optimal Single Dataset: {svmach_ac[i]}')

    tosay.append(f'Linear SVM Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(svmach_ac)}\n')
    print(f'Linear SVM Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(svmach_ac)}')
    
    for i in range(10):
        tosay.append(f'K-NN Test {i+1} Accuracy on Noisy Optimal Single Dataset: {knne_ac[i]}\n')
        print(f'K-NN Test {i+1} Accuracy on Noisy Optimal Single Dataset: {knne_ac[i]}')

    tosay.append(f'K-NN Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(knne_ac)}\n')
    print(f'K-NN Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(knne_ac)}')

    for i in range(10):
        tosay.append(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Optimal Single Dataset: {mlpe_ac[i]}\n')
        print(f'Multilayer Perceptron Test {i+1} Accuracy on Noisy Optimal Single Dataset: {mlpe_ac[i]}')

    tosay.append(f'Multilayer Perceptron Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(mlpe_ac)}\n')
    print(f'Multilayer Perceptron Test Average Accuracy on Noisy Optimal Single Dataset: {numpy.average(mlpe_ac)}')

    file_write(r"../Output/Single_Classifier_Noisy_Accuracy.txt", tosay)