import numpy as np

def activation(x):
    if x > 1:
        f = 1
    else:
        f = 0
    return f

def perceptron(X,d,alfa=0.05,max_it=100):
    X = np.insert(X,0,1,axis=1)

    W = np.zeros(X.shape[1])

    t = 1
    E = 1

    N = X.shape[0]
    y = np.zeros(N)

    while t < max_it and E > 0:
        E = 0
        for i in range(N):
            y[i] = activation(np.dot(W,X[i]))

            # Determina o erro
            error = d[i] - y[i]
            # Atualiza o vetor de Pesos
            W = W + np.dot(alfa*error,X[i])

            #Acumula o erro
            E = E + error**2

        t += 1
    return W

def use_W(W,x):
    x = np.insert(x,0,1)

    a = np.sum([W[i]*x[i] for i in range(W.shape[0])])

    return a

X = np.array([[1,1,0,1],[0,0,1,0],[1,1,0,0],[1,0,1,1],[1,0,0,1],[0,0,1,1]])

d = np.array([1,0,0,1,0,1])

W = perceptron(X,d)

print('Matrix de peso final:', W)

print(activation(use_W(W,[0,0,0,0])))
print(activation(use_W(W,[1,1,1,1])))