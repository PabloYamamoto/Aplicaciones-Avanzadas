import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# ==================================== Preprocesamiento de texto ====================================
def preprocesar_texto(lista_textos):
    lista_textos_preprocesados = []
    for texto in lista_textos:
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        texto = re.sub(r'\d+', '', texto)
        texto = re.sub(r'\s+', ' ', texto)
        lista_textos_preprocesados.append(texto)
    return lista_textos_preprocesados

# Carga los datos de entrenamiento
datos_originales = pd.read_csv('textos_originales.csv')
datos_plagiados = pd.read_csv('textos_plagiados.csv')

lista_textos_originales = datos_originales['texto'].tolist()
lista_textos_plagiados = datos_plagiados['texto'].tolist()

# Preprocesamiento de los datos
datos_preprocesados_originales = preprocesar_texto(lista_textos_originales)
datos_preprocesados_plagiados = preprocesar_texto(lista_textos_plagiados)


# ==================================== Extracción de características ====================================
# Creamos un objeto TfidfVectorizer() llamado vectorizador. 
# Este objeto nos permitirá transformar los documentos de texto en un conjunto de características(features) utilizando el método TF-IDF.
vectorizador = TfidfVectorizer()

# Usamos el método fit_transform() del objeto vectorizador para ajustar el modelo a los datos de entrenamiento y extraer las características de los textos originales. 
# El método fit_transform() realiza dos pasos: calcula los valores IDF y transforma los documentos en una matriz dispersa de características TF-IDF.
caracteristicas_originales = vectorizador.fit_transform(datos_preprocesados_originales)
caracteristicas_plagiados = vectorizador.transform(datos_plagiados['texto'])

# Esta variable es una matriz NumPy dispersa que contiene las características de todos los documentos. Cada fila de la matriz representa un documento de texto y cada columna representa una característica.
caracteristicas = np.concatenate((caracteristicas_originales.toarray(), caracteristicas_plagiados.toarray()), axis=0)


# ==================================== Selección de características ====================================
selector = SelectKBest(f_classif, k='all')
caracteristicas_seleccionadas = selector.fit_transform(caracteristicas, np.concatenate((np.zeros(datos_originales.shape[0]), np.ones(datos_plagiados.shape[0])), axis=0))


# ==================================== Entrenamiento del modelo ====================================
# División de los datos en conjuntos de entrenamiento y validación
X_originales = caracteristicas_originales.toarray()
# Etiquetas para los textos originales (0)
y_originales = np.zeros(len(datos_preprocesados_originales))
X_entrenamiento, X_validacion, y_entrenamiento, y_validacion = train_test_split(
    X_originales, y_originales, test_size=0.2, random_state=42)

# Entrenamiento del modelo SVM
clasificador_svm = SVC(kernel='linear')
clasificador_svm.fit(X_entrenamiento, y_entrenamiento)


