import numpy as np
from scipy.special import softmax

# ---------------------------------
# Funções
# ---------------------------------
def import_dataset(filepath,):
    file = open(filepath,'r')


    i,lines = 0, file.readlines()

    for line in lines:
        line = line.strip()

        sepal_lenght, sepal_width, petal_lenght, petal_width, classification = line.split(",")

def predict(X,W,i):
    y = np.zeros(3)
    y[0] = X[i, 0] + W[0, 0] * X[i, 1] + W[0, 1] * X[i, 2] + W[0, 2] * X[i, 3]
    y[1] = X[i, 0] + W[1, 0] * X[i, 1] + W[1, 1] * X[i, 2] + W[1, 2] * X[i, 3]
    y[2] = X[i, 0] + W[2, 0] * X[i, 1] + W[2, 1] * X[i, 2] + W[2, 2] * X[i, 3]

    y =softmax(y)
    print(y)
    ind = np.argmax(y)
    print(ind)

    return ind

