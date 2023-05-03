import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity as co
from sklearn.svm import SVC

# Leemos los archivos
with open('datos.txt', 'r') as file:
    archivos1 = file.readlines()
with open('datos2.txt', 'r') as file:
    archivos2 = file.readlines()

# Necesitamos tener los archivos en un solo arreglo, cada elemento del arreglo es un archivo
archivos = archivos1 + archivos2

vectorizer = CountVectorizer()
vectorizer = vectorizer.fit_transform(archivos)

similarity = co(vectorizer[0:1], vectorizer[1:2])
print("similitud:", similarity[0][0])

"""

vectorizer es una matriz dispersa, las matrices dispersas son una forma eficiente de representar matrices que tienen una gran cantidad de ceros. En lugar de alamacenar todos los valores cero, solo almeacenan los valores no cero y sus ubicaciones.

El primer número en los paréntesis (0, 32) es el indice del documento en tu conjunto de datos (en este caso, es la primera linea de texto o el primer documento)
El segundo número en los paréntesis es el indice del token o palabra en el vocabulario que se ha construido a partir de todos los documentos. 
El tercer número es el recuento de ese token en ese docuemtno

"""

# # Dividisión de los datos en entrenamiento y prueba
# x_train, x_test, y_train, y_test = train_test_split(
#     vectorizer, np.array([0, 1]), test_size=0.2)


# # Entrenamiento del modelo
# clf = SVC(kernel='linear', C=1, random_state=42)
#clf.fit(x_train, y_train)

# # Predicción y resultados
# ypred = clf.predict(x_test)
# print("Predicción: ", clf.score(x_test, y_test))
