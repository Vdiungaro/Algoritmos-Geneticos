# Vitor Diungaro - Perceptron

import numpy as np
import random
from scipy.special import softmax
import math


def perceptron(X_train,d_train,X_val,d_val,alfa,max_it):
    t = 1
    E = 1

    N = X_train.shape[0]

    n_class = d_train.shape[1]

    W = np.random.rand(n_class,X_train.shape[1])
    b = np.random.rand(n_class,N)

    y = np.zeros((n_class,N))
    e = np.zeros((n_class,N))

    E_train = np.array([])
    E_val = np.array([])

    while t < max_it and E > 0:
        E = 0
        for i in range(N):
            # Saida da rede
            y[0, i] = np.dot(W[0], X_train[i]) + b[0, i]
            y[1, i] = np.dot(W[1], X_train[i]) + b[1, i]
            y[2, i] = np.dot(W[2], X_train[i]) + b[2, i]
            y[:,i] = softmax(y[:,i])

            # Calcula o Erro
            e[0, i] = d_train[i, 0] - y[0, i]
            e[1, i] = d_train[i, 1] - y[1, i]
            e[2, i] = d_train[i, 2] - y[2, i]

            # Atualiza os vetores de Peso
            W[0] = W[0] + np.dot(alfa * e[0, i], X_train[i])
            W[1] = W[1] + np.dot(alfa * e[1, i], X_train[i])
            W[2] = W[2] + np.dot(alfa * e[2, i], X_train[i])

            # Atualiza o bias
            b[0, i] = b[0, i] + (alfa * e[0, i])
            b[1, i] = b[1, i] + (alfa * e[1, i])
            b[2, i] = b[2, i] + (alfa * e[2, i])

            # Salva o Erro
            E = E + ((math.pow(e[0,i],2) + math.pow(e[1,i],2) + math.pow(e[2,i],2))/3)

        # Reduz o erro caso chegue muito perto de zero
        if E < 0.01:
            if round(E,2) < 0.05:
                E = 0

        #print(E)

        E_train = np.append(E_train,E)

        aux = validation(X_val,d_val,W,b)
        E_val = np.append(E_val,aux)

        t += 1

    print('\nTreinamento Concluido!\n')

    return W,b,E_train,E_val

# Validação no fim de cada epoca
def validation(X_val,d_val,W,b):
    N = X_val.shape[0]
    n_class = d_val.shape[1]

    y = np.zeros((n_class, N))
    e = np.zeros((n_class, N))

    E_val = 0.0

    for i in range(N):
        # Saida da rede
        y[0, i] = np.dot(W[0], X_val[i]) + b[0, i]
        y[1, i] = np.dot(W[1], X_val[i]) + b[1, i]
        y[2, i] = np.dot(W[2], X_val[i]) + b[2, i]
        y[:, i] = softmax(y[:, i])

        # Calcula o erro
        e[0, i] = d_val[i, 0] - y[0, i]
        e[1, i] = d_val[i, 1] - y[1, i]
        e[2, i] = d_val[i, 2] - y[2, i]

        # Salva o erro
        E_val = E_val + ((math.pow(e[0, i], 2) + math.pow(e[1, i], 2) + math.pow(e[2, i], 2)) / 3)

    # Reduz o erro caso chegue muito perto de zero
    if E_val <0.01:
        if round(E_val,2) < 0.05:
            E_val = 0

    return E_val
