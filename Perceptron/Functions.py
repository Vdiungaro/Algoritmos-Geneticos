# Vitor Diungaro - Funções gerais utilizadas

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
import random
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os

# Inicializa o dataframe Classificador de sentimentos
def import_dataframe(file_path):
    df = pd.read_csv(file_path, header=None, sep='\t', names=['Sentences', 'Score'])

    return df

# Junta os dataframes em um só
def merge_dataframe(df1, df2, df3):
    frames = [df1, df2, df3]
    df = pd.concat(frames, ignore_index=True)

    return df

# Treina a rede
def training(train, vectorizer):
    Train_Sentences = vectorizer.fit_transform(train['Sentences'])
    Train_Score = train['Score']
    return BernoulliNB().fit(Train_Sentences, Train_Score)

# Função para mostrar os resultados na tela
def show_results(value):
    phrase, result = value
    if result == 1:
        result = 'Positive Phrase'
    else:  # result == 0
        result = 'Negative Phrase'
    print(phrase, ':', result)

# Função que faz a analise de um comentario
def analysis(phrase, classifier, vectorizer):
    return phrase, classifier.predict(vectorizer.transform([phrase]))

# Testa a rede e retorna os valores de precisão da rede
def testing(test,classifier,vectorizer):
    Test_Sentences = test['Sentences']
    Test_Score = test['Score']
    Test_Sentences_ind = test.index

    total = len(Test_Sentences_ind)

    sucess = 0
    false_positive = 0
    true_positive = 0
    false_negative = 0
    true_negative = 0

    for i in range(total):
        j = random.choice(Test_Sentences_ind)
        test_result = analysis(Test_Sentences[j],classifier,vectorizer)
        phrase, result = test_result
        if result == Test_Score[j]:
            sucess += 1
        if result == 0 and Test_Score[j] == 0:
            true_negative += 1
        elif result == 0 and Test_Score[j] == 1:
            false_negative += 1
        elif result == 1 and Test_Score[j] == 1:
            true_positive += 1
        elif result == 1 and Test_Score[j] == 0:
            false_positive += 1

    return (sucess*100/total , false_positive*100/total, true_positive*100/total, false_negative*100/total, true_negative*100/total )

# Faz a avaliação da rede e retorna as classes reais e previstas para a matriz de confusão
def evaluation(test,classifier,vectorizer):
    Test_Sentences = test['Sentences']
    Test_Score = test['Score']
    Test_Sentences_ind = test.index

    total = len(Test_Sentences_ind)
    prediction = np.zeros(total)
    real_class = np.zeros(total)

    for i in range(total):
        j = random.choice(Test_Sentences_ind)
        test_result = analysis(Test_Sentences[j],classifier,vectorizer)
        phrase, result = test_result
        prediction[i] = result
        real_class[i] = Test_Score[j]

    return prediction,real_class

# -----------------------------------------------
# Funções para gerar graficos
# -----------------------------------------------

def plot(E_train,E_val,file_path):
    figure_2 = plt.figure()
    plt.plot(E_train, label='Treinamento')
    plt.plot(E_val,label='Validação')
    plt.xlabel('Epoca')
    plt.ylabel('Erro')
    plt.title('Erro durante o Treinamento')
    plt.legend()
    figure_2.savefig(str(file_path))
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classes, type_of_data, file_path):
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm)
    print(df)
    ax = sns.heatmap(df,annot=True,cmap='YlGnBu',xticklabels=classes,yticklabels=classes,cbar = 'optional',fmt='d')
    ax.set_title("Matriz de confusão do " + str(type_of_data+"."))
    ax.set(xlabel='Previsto', ylabel='Verdadeira')
    fig = ax.get_figure()
    fig.savefig(str(file_path),bbox_inches='tight')
    ax.get_figure().clf()


