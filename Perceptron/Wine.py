# Vitor Diungaro - Wine Dataset
import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from scipy.stats import stats
import Perceptron
import Functions

# --------------------------------
# Funções
# --------------------------------
# Obtem os dados do dataset e os divide em treino, teste e validação
def import_dataset(filepath,number_samples,number_attributes,number_class):
    file = open(filepath,'r')

    X = np.zeros((number_samples,number_attributes))
    d = np.zeros((number_samples,number_class))

    i, lines = 0, file.readlines()

    for line in lines:
        line = line.strip()  # remove \n

        classification, alcohol, malic_acid, ash, ash_alcalinity, magnesium, phenols, flavanoids, non_flav_phenols, protoan, color_int, hue, od, proline = line.split(",")

        X[i] = np.array([alcohol, malic_acid, ash, ash_alcalinity, magnesium, phenols, flavanoids, non_flav_phenols, protoan, color_int, hue, od, proline])

        if classification == '1':
            d[i] = np.array([1,0,0])
        elif classification == '2':
            d[i] = np.array([0,1,0])
        else:
            d[i] = np.array([0,0,1])

        i += 1
    file.close()

    # Normaliza com zscore
    X = stats.zscore(X, axis=0)

    # Separa em treino, teste e validação
    X_train, X_val, d_train, d_val = train_test_split(X, d, test_size=0.4, shuffle=True)
    X_val, X_test, d_val, d_test = train_test_split(X_val, d_val, test_size=0.5, shuffle=True)

    return X_train, X_val, X_test, d_train, d_val, d_test

# Retorna a Classe de um item
def predict(X,W,b,i):
    y = np.zeros(3)

    y[0] = np.sum([W[0, j] * X[i, j] for j in range(W.shape[1])]) + b[0, i]
    y[1] = np.sum([W[1, j] * X[i, j] for j in range(W.shape[1])]) + b[1, i]
    y[2] = np.sum([W[2, j] * X[i, j] for j in range(W.shape[1])]) + b[2, i]

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
    X_train, X_val, X_test, d_train, d_val, d_test = import_dataset(filepath='Datasets/Ex2/wine.data',
                                                                    number_samples=178, number_attributes=13,
                                                                    number_class=3)
    # Treina a rede
    W, b, E_train, E_val = Perceptron.perceptron(X_train, d_train, X_val, d_val, alfa=0.007, max_it=500)
    Functions.plot(E_train, E_val, file_path='Results/Wine/erro.pdf')

    # Avaliação
    pred_test, true_test = evaluation(X_test, d_test, W, b)
    Functions.plot_confusion_matrix(true_test, pred_test, classes=['Wine 1', 'Wine 2', 'Wine 3'],
                                    type_of_data="conjunto de testes", file_path="Results/Wine/Matrixteste.pdf")

    pred_train, true_train = evaluation(X_train, d_train, W, b)
    Functions.plot_confusion_matrix(true_train, pred_train, classes=['Wine 1', 'Wine 2', 'Wine 3'],
                                    type_of_data="conjunto de treino", file_path="Results/Wine/Matrixtreino.pdf")

    pred_val, true_val = evaluation(X_val, d_val, W, b)
    Functions.plot_confusion_matrix(true_val, pred_val, classes=['Wine 1', 'Wine 2', 'Wine 3'],
                                    type_of_data="conjunto de validação", file_path="Results/Wine/Matrixvalidação.pdf")

if __name__ == '__main__':
    main()



