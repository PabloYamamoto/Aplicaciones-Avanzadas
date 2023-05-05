import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
import joblib
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as co
import matplotlib.pyplot as plt

# ==================================== Preprocesamiento de texto ====================================
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

# Compute the normalized LCS given an answer text and a source text
def lcs_norm_word(answer_text, source_text):
    '''Computes the longest common subsequence of words in two texts; returns a normalized value.
       :param answer_text: The pre-processed text for an answer text
       :param source_text: The pre-processed text for an answer's associated source text
       :return: A normalized LCS value'''
    
    a_text = answer_text.split()
    s_text = source_text.split()
    
    n = len(a_text)
    m = len(s_text)
    
    # create an m x n matrix
    matrix_lcs = np.zeros((m+1,n+1), dtype=int)
    
    # iterate through each word in the source text looking for a match against the answer text
    for i, s_word in enumerate(s_text, start=1):
        for j, a_word in enumerate(a_text, start=1):
            # match: diagonal addition
            if a_word == s_word:
                matrix_lcs[i][j] = matrix_lcs[i-1][j-1] + 1
            else:
            # no match: max of top/left values
                matrix_lcs[i][j] = max(matrix_lcs[i-1][j], matrix_lcs[i][j-1])
    
    # normalize lcs = (last value in the m x n matrix) / (length of the answer text)
    normalized_lcs = matrix_lcs[m][n] / n

    return normalized_lcs

def create_lcs_features(df):
    
    lcs_values = []
    
    # iterate through files in dataframe
    for i in df.index:
        # calculate lcs
        lcs = lcs_norm_word(df["texto2"][i], df["texto1"][i])
        lcs_values.append(lcs)

    print('LCS features created!')
    return lcs_values


def get_cos_sim(df):
    datos = pd.concat([df["texto1"], df["texto2"]])
    vector1 = CountVectorizer().fit_transform(datos)
    cos_sim = []
    for i in df.index:
        # calculate cosine similarity
        cos = co(vector1[i], vector1[i+n])
        cos_sim.append(cos)

    print('Cosine similarity features created!')
    return cos_sim

def grafh_df(df):
    groups = df.groupby('etiqueta')

    for name, group in groups:
        plt.plot(group['lcs_values'], group['cos_sim'], marker='o', linestyle='', markersize=12, label=name)

    # Crear la gráfica de puntos con diferentes colores para cada valor de la columna 'z'
    # plt.scatter(df['lcs_values'], df['cos_sim'] , c=df['etiqueta'])

    # Añadir etiquetas y título a la gráfica
    plt.legend()
    plt.xlabel('lcs values')
    plt.ylabel('cos values')
    plt.title('Gráfica de puntos con diferencia de color de plagio o no.')

    # Mostrar la gráfica
    plt.show()
    pass

n = 0
with open("Data/train_snli.txt", 'r') as file:
    data  = file.read().lower().split("\n")

if n == 0:
    n = len(data)
data = data[:n]
data = list(map(lambda data: data.split("\t"), data))

# Crea dos listas separadas para los datos y etiquetas
datos = [[fila[0], fila[1]] for fila in data]
etiquetas = [fila[2] for fila in data]

# Crea el DataFrame a partir de los datos y etiquetas
df = pd.DataFrame(datos, columns=['texto1', 'texto2'])
df['etiqueta'] = etiquetas

df["lcs_values"] = create_lcs_features(df)
df["cos_sim"] = get_cos_sim(df)
# grafh_df(df)
# print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[['lcs_values', 'cos_sim']], df[['etiqueta']], test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = svm.SVC()

## DONE: Train the model
model.fit(X_train, y_train)
# Save the trained model
joblib.dump(model,  "model2.pkl")

y_predict = model.predict(X_test)
acc = model.score(X_test, y_test)
print(acc)