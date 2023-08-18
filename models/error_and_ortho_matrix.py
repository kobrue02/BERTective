"""
Wir testen mit diesem Skript, inwiefern sich die Verwendung von bestimmten Orthographie-Versionen
und bestimmten Fehlern dazu eignen, Aussagen aber den Bildungsgrad eines Autoren zu treffen.
Für die Texten werden mithilfe von Wortlisten, die wir vom Rechtschreibforum https://www.korrekturen.de/ beziehen,
"Fehlervektoren" und "Orthographievektoren" erzeugt, welche wir zum Trainieren und Testen verschiedener Klassifizierungs-
modelle verwenden. Die Ergebnisse sind vielversprechend.
"""

import json
import numpy as np
import pandas as pd
import spacy

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

nlp = spacy.load("de_core_news_sm")

with open("../data/reddit/education/Azubis.json", "r", encoding='utf-8') as f:
    azubis_data = json.load(f)

with open("../data/reddit/education/Studium.json", "r", encoding='utf-8') as f:
    studi_data = json.load(f)

with open("../data/reddit/dating/beziehungen.json", "r", encoding='utf-8') as f:
    bzh_data = json.load(f)

with open("../data/annotation/orthography.json", "r", encoding='utf-8') as f:
    orthography = json.load(f)

with open("../data/annotation/error_tuples.json", "r", encoding='utf-8') as f:
    error_tuples = json.load(f)


def vector_average(vectors: list) -> np.array:
    result = np.zeros((96,))
    for vector in vectors:
        result = np.add(result, vector)
    return np.divide(result, len(vectors))

def find_ortho_match_in_text(text: str, orthography_set: str) -> np.array:
    '''
    generate word embeddings for matches with a given orthography set
        Parameters:
            text: a text sample
            orthography_set: a list from the orthography json
        Returns:
            the averaged vector of the matched embeddings
    '''
    matched_vectors = []
    for item in orthography["orthographies"][orthography_set]:
        if item in text and item != "":
            matched_vectors.append(nlp(str(item)).vector)  # SpaCy tok2vec
    if matched_vectors == []:
        return np.zeros((96,))
    else:
        # average the embeddings
        return vector_average(matched_vectors)
    

def find_error_match_in_text(text: str, reference_set: str) -> np.array:
    '''
    generate word embeddings for matches with a common spelling error list
        Parameters:
            text: a text sample
            orthography_set: a list from the orthography json
        Returns:
            the averaged vector of the matched embeddings
    '''
    matched_vectors = []

    for i in range(len(error_tuples["errors"])):
        # either check errors or correct spellings
        if reference_set == "error":
            item = error_tuples["errors"][i][0]
        elif reference_set == "correct":
            item = error_tuples["errors"][i][1]
        else:
            raise TypeError("reference has to be 'error' or 'correct'")
        # get embedding
        if item in text and item != "":
            matched_vectors.append(nlp(str(item)).vector)
    if matched_vectors == []:
        return np.zeros((96,))
    else:
        # average the embeddings
        return vector_average(matched_vectors)


reddit_matrix_list = []

for item in azubis_data["data"]:
    tup = ("Azubi", str(item["selftext"]))
    reddit_matrix_list.append(tup)

for item in studi_data["data"]:
    tup = ("Student", str(item["selftext"]))
    reddit_matrix_list.append(tup)

#for item in bzh_data["data"]:
#    tup = ("Beziehungen", str(item["selftext"]))
#    reddit_matrix_list.append(tup)

reddit_matrix = pd.DataFrame()

reddit_matrix["Source"] = [i[0] for i in reddit_matrix_list]
reddit_matrix["Text"] = [i[1] for i in reddit_matrix_list]

reddit_matrix["AncOrthoVec"] = [find_ortho_match_in_text(i[1], "ancient") for i in reddit_matrix_list]
reddit_matrix["AncOrthoVec"] = reddit_matrix["AncOrthoVec"].squeeze()

reddit_matrix["RevOrthoVec"] = [find_ortho_match_in_text(i[1], "revolutionized") for i in reddit_matrix_list]
reddit_matrix["RevOrthoVec"] = reddit_matrix["RevOrthoVec"].squeeze()

reddit_matrix["ModOrthoVec"] = [find_ortho_match_in_text(i[1], "modern") for i in reddit_matrix_list]
reddit_matrix["ModOrthoVec"] = reddit_matrix["ModOrthoVec"].squeeze()

reddit_matrix["ErrorVec"] = [find_error_match_in_text(i[1], "error") for i in reddit_matrix_list]
reddit_matrix["ErrorVec"] = reddit_matrix["ErrorVec"].squeeze()

reddit_matrix["CorrectSpellingVec"] = [find_error_match_in_text(i[1], "correct") for i in reddit_matrix_list]
reddit_matrix["CorrectSpellingVec"] = reddit_matrix["CorrectSpellingVec"].squeeze()

reddit_matrix["Compiled_Orthography"] = reddit_matrix["AncOrthoVec"] + \
                                        reddit_matrix["ModOrthoVec"] + \
                                        reddit_matrix["RevOrthoVec"] + \
                                        reddit_matrix["CorrectSpellingVec"] + \
                                        reddit_matrix["ErrorVec"]


if __name__ == "__main__":
    model = LogisticRegression(max_iter=10000)

    X = reddit_matrix["Compiled_Orthography"].tolist()
    y = reddit_matrix["Source"].tolist()

    X, y = shuffle(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))


    """
                 precision    recall  f1-score   support

       Azubi       0.74      0.93      0.83       140
     Student       0.88      0.62      0.73       120

    accuracy                            0.79       260
    macro avg       0.81      0.78      0.78       260
    weighted avg    0.81      0.79      0.78       260

    """