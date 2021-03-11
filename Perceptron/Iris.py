# Vitor Diungaro - Iris Dataset

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
import Perceptron
import Functions
# ---------------------------------
# Funções
# ---------------------------------
# Obtem os dados do dataset e os divide em treino, teste e validação
def import_dataset(filepath,number_samples,number_attributes,number_class):
    file = open(filepath,'r')

    X = np.zeros((number_samples,number_attributes))
    d = np.zeros((number_samples,number_class))

    i, lines = 0, file.readlines()

    for line in lines:
        line = line.strip()  # remove \n

        sepal_lenght, sepal_width, petal_lenght, petal_width, classification = line.split(',')
        X[i] = np.array([sepal_lenght, sepal_width, petal_lenght, petal_width])

        if classification == "Iris-setosa":
            d[i] = np.array([1,0,0])
        elif classification == 'Iris-versicolor':
            d[i] = np.array([0,1,0])
        else:
            d[i] = np.array([0,0,1])
        i+=1

    file.close()
    # Separa em treino, teste e validação
    X_train, X_val, d_train, d_val = train_test_split(X,d, test_size=0.4, shuffle=True)
    X_val, X_test, d_val, d_test = train_test_split(X_val,d_val, test_size=0.5, shuffle=True)

    return X_train, X_val, X_test, d_train, d_val, d_test

# Retorna a Classe de um item
def predict(X,W,b,i):
    y = np.zeros(3)
    y[0] = W[0, 0] * X[i, 0] + W[0, 1] * X[i, 1] + W[0, 2] * X[i, 2] + W[0, 3] * X[i, 3] + b[0, i]
    y[1] = W[1, 0] * X[i, 0] + W[1, 1] * X[i, 1] + W[1, 2] * X[i, 2] + W[1, 3] * X[i, 3] + b[1, i]
    y[2] = W[2, 0] * X[i, 0] + W[2, 1] * X[i, 1] + W[2, 2] * X[i, 2] + W[2, 3] * X[i, 3] + b[2, i]

    # Gera Y normalizado ex:[0,0,1]
    y = softmax(y)

    ind = np.argmax(y)

    return ind

# Faz a avaliação da rede e retorna as classes previstas e as classes reais para a matriz de confusão
def evaluation(X,d,W,b,):
    N = X.shape[0]
    acerto = 0

    prediction = np.zeros(N)
    real_class = np.zeros(N)

    # Para cada entrada
    for i in range(N):
        # Faz uma predição com W e b
        predicted = predict(X,W,b,i)
        true_class = np.argmax(d[i])

        prediction[i] = predicted
        real_class[i] = true_class

        # Verifica se foi um acerto
        if true_class == predicted:
            acerto += 1

    print("Quantidade de entradas no conjunto: ", + N)
    print("Quantidade de acertos: ", + acerto)
    print("Quantidade de erros: ", + N-acerto)
    print("Taxa de acerto: %.2f%%", + (acerto/N * 100))
    print("Taxa de erro: %.2f%%", + ((N-acerto)/N * 100), "\n")

    return prediction,real_class


def main():
    # Inicializa o dataset
    X_train, X_val, X_test, d_train, d_val, d_test = import_dataset(filepath='Datasets/Ex1/iris.data', number_samples=150,number_attributes=4,number_class=3)

    # Treina a rede
    W, b, E_train, E_val = Perceptron.perceptron(X_train, d_train,X_val,d_val, alfa=0.002, max_it=500)
    Functions.plot(E_train,E_val,file_path='Results/Iris/erro.pdf')

    # Avaliação
    pred_test, true_test = evaluation(X_test,d_test,W,b)
    Functions.plot_confusion_matrix(true_test,pred_test,classes=['I. setosa', 'I. versicolor', 'I. virginica'],
                                    type_of_data="conjunto de testes",file_path="Results/Iris/Matrixteste.pdf")

    pred_train, true_train = evaluation(X_train,d_train,W,b)
    Functions.plot_confusion_matrix(true_train, pred_train, classes=['I. setosa', 'I. versicolor', 'I. virginica'],
                                    type_of_data="conjunto de treino", file_path="Results/Iris/Matrixtreino.pdf")

    pred_val, true_val = evaluation(X_val,d_val,W,b)
    Functions.plot_confusion_matrix(true_val, pred_val, classes=['I. setosa', 'I. versicolor', 'I. virginica'],
                                    type_of_data="conjunto de validação", file_path="Results/Iris/Matrixvalidação.pdf")

if __name__ == '__main__':
    main()