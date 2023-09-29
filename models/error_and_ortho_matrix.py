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

class OrthoMatrixModel:

    def __init__(self, use_model = None, ex: bool = False) -> None:

        """ 
        This class is used to do a matching between text samples and data which was 
        sourced from https://www.korrekturen.de/
        """

        with open("data/reddit/dating/beziehungen.json", "r", encoding='utf-8') as f:
            self.bzh_data = json.load(f)
        with open("data/annotation/orthography.json", "r", encoding='utf-8') as f:
            self.orthography = json.load(f)
        with open("data/annotation/error_tuples.json", "r", encoding='utf-8') as f:
            self.error_tuples = json.load(f)
        
        if ex:

            # TO  BE CLEANED UP LATER
            with open("data/reddit/education/Azubis.json", "r", encoding='utf-8') as f:
                self.azubis_data = json.load(f)
            with open("data/reddit/education/Studium.json", "r", encoding='utf-8') as f:
                self.studi_data = json.load(f)

            reddit_matrix_list = []

            for item in self.azubis_data["data"]:
                tup = ("Azubi", str(item["selftext"]))
                reddit_matrix_list.append(tup)

            for item in self.studi_data["data"]:
                tup = ("Student", str(item["selftext"]))
                reddit_matrix_list.append(tup)

            self.reddit_matrix = self.__build_reddit_matrix(reddit_matrix_list)
            self.use_model = use_model
            self.model = self.__build_model()


    def __vector_average(self, vectors: list) -> np.array:
        result = np.zeros((96,))
        for vector in vectors:
            result = np.add(result, vector)
        return result

    def find_ortho_match_in_text(self, text: str, orthography_set: str) -> np.ndarray:
        '''
        generate word embeddings for matches with a given orthography set
            Parameters:
                text: a text sample
                orthography_set: a list from the orthography json
            Returns:
                the averaged vector of the matched embeddings
        '''
        matched_vectors = []
        for item in self.orthography["orthographies"][orthography_set]:
            if item in text and item != "":
                matched_vectors.append(nlp(str(item)).vector)  # SpaCy tok2vec
        if matched_vectors == []:
            return np.zeros((96,))
        else:
            # average the embeddings
            return self.__vector_average(matched_vectors)
    

    def find_error_match_in_text(self, text: str, reference_set: str) -> np.ndarray:
        '''
        generate word embeddings for matches with a common spelling error list
            Parameters:
                text: a text sample
                orthography_set: a list from the orthography json
            Returns:
                the averaged vector of the matched embeddings
        '''
        matched_vectors = []

        for i in range(len(self.error_tuples["errors"])):
            # either check errors or correct spellings
            if reference_set == "error":
                item = self.error_tuples["errors"][i][0]
            elif reference_set == "correct":
                item = self.error_tuples["errors"][i][1]
            else:
                raise TypeError("reference has to be 'error' or 'correct'")
            # get embedding
            if item in text and item != "":
                matched_vectors.append(nlp(str(item)).vector)
        if matched_vectors == []:
            return np.zeros((96,))
        else:
            # average the embeddings
            return self.__vector_average(matched_vectors)
    
    def __build_reddit_matrix(self, text_list: list) -> pd.DataFrame:
        reddit_matrix = pd.DataFrame()

        reddit_matrix["Source"] = [i[0] for i in text_list]
        reddit_matrix["Text"] = [i[1] for i in text_list]

        reddit_matrix["AncOrthoVec"] = [self.find_ortho_match_in_text(i[1], "ancient") for i in text_list]
        reddit_matrix["AncOrthoVec"] = reddit_matrix["AncOrthoVec"].squeeze()

        reddit_matrix["RevOrthoVec"] = [self.find_ortho_match_in_text(i[1], "revolutionized") for i in text_list]
        reddit_matrix["RevOrthoVec"] = reddit_matrix["RevOrthoVec"].squeeze()

        reddit_matrix["ModOrthoVec"] = [self.find_ortho_match_in_text(i[1], "modern") for i in text_list]
        reddit_matrix["ModOrthoVec"] = reddit_matrix["ModOrthoVec"].squeeze()

        reddit_matrix["ErrorVec"] = [self.find_error_match_in_text(i[1], "error") for i in text_list]
        reddit_matrix["ErrorVec"] = reddit_matrix["ErrorVec"].squeeze()

        reddit_matrix["CorrectSpellingVec"] = [self.find_error_match_in_text(i[1], "correct") for i in text_list]
        reddit_matrix["CorrectSpellingVec"] = reddit_matrix["CorrectSpellingVec"].squeeze()

        reddit_matrix["Compiled_Orthography"] = reddit_matrix["AncOrthoVec"] + \
                                                reddit_matrix["ModOrthoVec"] + \
                                                reddit_matrix["RevOrthoVec"] + \
                                                reddit_matrix["CorrectSpellingVec"] + \
                                                reddit_matrix["ErrorVec"]
        
        return reddit_matrix
    
    def __build_model(self):

        model = self.use_model

        X = self.reddit_matrix["Compiled_Orthography"].tolist()
        y = self.reddit_matrix["Source"].tolist()

        X, y = shuffle(X, y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

        model.fit(self.X_train, self.y_train)
        return model
    
    def evaluate(self):

        y_pred = self.model.predict(self.X_test)

        print(classification_report(self.y_test, y_pred))


if __name__ == "__main__":

    import os
    print(os.path.abspath(os.curdir))
    os.chdir("..")

    test = OrthoMatrixModel()
    sample = """Wir wollten ursprünglich mehr Blogs sourcen, aber keine guten gefunden: entweder nur wenige Autoren (unnötig) oder anonyme Autoren (bringt nichts) -> muss viele Autoren mit Lebenslauf/Beschreibung geben (Achgut)"""

    ancient = test.find_ortho_match_in_text(sample, "ancient")
    revolutionized = test.find_ortho_match_in_text(sample, "revolutionized")
    modern = test.find_ortho_match_in_text(sample, "modern")
    error = test.find_error_match_in_text(sample, "error")
    correct = test.find_error_match_in_text(sample, "correct")
    print(ancient)
    print("-"*64)
    print(revolutionized)
    print("-"*64)
    print(modern)
    print("-"*64)
    print(error)
    print("-"*64)
    print(correct)

    exit()

    model = OrthoMatrixModel(
        use_model=LogisticRegression(max_iter=10000)
        )

    model.evaluate()

    """
                  precision    recall  f1-score   support

       Azubi       0.76      0.98      0.85       140
     Student       0.96      0.63      0.76       120

    accuracy                           0.82       260
   macro avg       0.86      0.81      0.81       260
weighted avg       0.85      0.82      0.81       260

    """