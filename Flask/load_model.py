import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as co

model = joblib.load("../trained_models/SVM_2.pkl")

def lcs_norm_word(answer_text: str, source_text: str):
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

def get_cos_sim(answer_text: str, source_text: str):
    datos = [answer_text, source_text]
    vector1 = CountVectorizer().fit_transform(datos)

    cos = co(vector1[0], vector1[1])

    return cos

def get_features(answer_text: str, source_text: str):
    lcs = lcs_norm_word(answer_text, source_text)
    cos = get_cos_sim(answer_text, source_text)

    return (lcs, cos)

def predict(answer_text, source_text):
    lcs, cos = get_features(answer_text, source_text)

    features = [lcs, cos]
    print(features)

    return model.predict([features])[0]

print(predict("I am a student who want to pass the exam", "I am a student who study in a university"))