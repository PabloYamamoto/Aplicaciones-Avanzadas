import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif

# Preprocesamiento de texto
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

print(datos_originales)
print(datos_plagiados)

lista_textos_originales = datos_originales['texto'].tolist()
lista_textos_plagiados = datos_plagiados['texto'].tolist()

print(lista_textos_originales)
print(lista_textos_plagiados)


# datos_preprocesados_originales = preprocesar_texto(lista_textos_originales)
# datos_preprocesados_plagiados = preprocesar_texto(lista_textos_plagiados)


# # Extracción de características
# vectorizador = TfidfVectorizer()
# caracteristicas_originales = vectorizador.fit_transform(
#     datos_originales['texto'])
# caracteristicas_plagiados = vectorizador.transform(datos_plagiados['texto'])
# caracteristicas = np.concatenate(
#     (caracteristicas_originales.toarray(), caracteristicas_plagiados.toarray()), axis=0)

# # Imprime las características extraídas para los primeros tres textos originales
# print("\nCaracterísticas extraídas para los primeros tres textos originales:")
# print(caracteristicas_originales.toarray()[:3])

# # Imprime las características extraídas para los primeros tres textos plagiados
# print("\nCaracterísticas extraídas para los primeros tres textos plagiados:")
# print(caracteristicas_plagiados.toarray()[:3])

# # Selección de características
# selector = SelectKBest(f_classif, k=1000)
# caracteristicas_seleccionadas = selector.fit_transform(caracteristicas, np.concatenate(
#     (np.zeros(datos_originales.shape[0]), np.ones(datos_plagiados.shape[0])), axis=0))

# # Imprime las 10 características más relevantes
# nombres_caracteristicas = vectorizador.get_feature_names()
# scores_caracteristicas = selector.scores_
# scores_caracteristicas /= np.max(scores_caracteristicas)
# indices_top = np.argsort(scores_caracteristicas)[::-1][:10]
# print("\nLas 10 características más relevantes:")
# for i in indices_top:
#     print(f"{nombres_caracteristicas[i]} ({scores_caracteristicas[i]:.2f})")
