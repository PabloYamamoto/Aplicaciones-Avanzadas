import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn import svm

# ==================================== Preprocesamiento de texto ====================================
def preprocesar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

# Carga los datos de entrenamiento
# dtypes = {'File_index': str, 'Plagiarism': int}
# datos = pd.read_csv('Plagiarism_Detection_DS/input/Plagiarized.csv', dtype=dtypes)
# textos = []
# for i in datos['File_index']:
#     with open('Plagiarism_Detection_DS/input/suspicious-document{}.txt'.format(i), 'r') as file:
#         textos.append(preprocesar_texto(file.read()))

# datos["Textos"] = textos

# datos.to_csv('Plagiarized_preprocesado.csv', index=False)

# with open("Plagiarized_preprocesado.csv", 'r') as file:
#     datos = file.read()

with open("train_snli.txt", 'r') as file:
    data  = file.read().lower().split("\n")
data = list(map(lambda data: data.split("\t"), data))

# Crea dos listas separadas para los datos y etiquetas
datos = [[fila[0], fila[1]] for fila in data]
etiquetas = [fila[2] for fila in data]

# Crea el DataFrame a partir de los datos y etiquetas
df = pd.DataFrame(datos, columns=['texto1', 'texto2'])
df['etiqueta'] = etiquetas

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

df["lcs_values"] = create_lcs_features(df)

print(df.head())

X_train, X_test, y_train, y_test = train_test_split(df[['lcs_values']], df[['etiqueta']], test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
model = svm.SVC()
    
    
## DONE: Train the model
model.fit(X_train, y_train)

y_predict = model.predict(X_test)

print(y_predict)
print(y_test)