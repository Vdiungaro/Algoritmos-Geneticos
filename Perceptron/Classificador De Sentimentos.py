# Vitor Diungaro - Classificador de sentimentos
import pandas as pd
import Functions
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Gera dataframes necessarios
df_amazon = Functions.import_dataframe(file_path='Datasets/Ex3/amazon_cells_labelled.txt')
df_imdb = Functions.import_dataframe(file_path='Datasets/Ex3/imdb_labelled.txt')
df_yelp = Functions.import_dataframe(file_path='Datasets/Ex3/yelp_labelled.txt')

# Transforma os dataframes em um só
df = Functions.merge_dataframe(df_amazon, df_imdb, df_yelp)
# Salva o dataframe em um arquivo txt
df.to_csv('Datasets/Ex3/dfcomplete.txt')
# print(df)

# Divide em conjunto de treino e teste
train, test = train_test_split(df, test_size=0.25)
# Inicia o vetorizador
vectorizer = CountVectorizer(binary='True')
# Treina o classificador
classifier = Functions.training(train, vectorizer)

#Testa uma frase do usuario
Functions.show_results(Functions.analysis('I hate this movie', classifier, vectorizer))

# Obtem as estatisticas
precision, false_positive, true_positive, false_negative, true_negative = Functions.testing(test,classifier,vectorizer)

print('The model precision is:', precision, '%')
print('With', true_positive, '% being true positives')
print('With', false_positive, '% being false positives')
print('With', true_negative, '% being true negatives')
print('With', false_negative, '% being false negatives')

# Avaliação e matriz de confusão
pred_test,true_test = Functions.evaluation(test,classifier,vectorizer)
Functions.plot_confusion_matrix(pred_test,true_test,classes=['Positivo','Negativo'],type_of_data='conjunto de teste',file_path='Results/Ex3/Matrizteste.pdf')

pred_train,true_train = Functions.evaluation(train,classifier,vectorizer)
Functions.plot_confusion_matrix(pred_train,true_train,classes=['Positivo','Negativo'],type_of_data='conjunto de treino',file_path='Results/Ex3/Matriztreino.pdf')