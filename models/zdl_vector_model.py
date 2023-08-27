# to import scraper from same-level subdirectory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from corpus import DataCorpus, DataObject
from german_stopwords import stopwords_
from scraping_tools.zdl_regio_client import zdl_request, tokenize

import json
from langdetect import detect, DetectorFactory, lang_detect_exception
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import requests

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz

import seaborn as sns
from tqdm import tqdm

AREAL_DICT = {
              "DE-NORTH-EAST": ["Rostock", "Berlin","luebeck", "Potsdam"],
              "DE-NORTH-WEST": ["bremen", "Hamburg","Hannover", "bielefeld", "Dortmund", "kiel", "Paderborn"],
              "DE-MIDDLE-EAST": ["Leipzig", "dresden", "HalleSaale", "erfurt", "jena"],
              "DE-MIDDLE-WEST": ["frankfurt", "duesseldorf", "cologne","Bonn", "Mainz", "Wiesbaden", "kaiserslautern", "Mannheim", "Saarland"],
              "DE-SOUTH-EAST": ["Munich", "Nurnberg", "Regensburg", "Würzburg", "bayreuth", "bavaria"],
              "DE-SOUTH-WEST": ["stuttgart", "augsburg", "freiburg", "karlsruhe", "Ulm", "Tuebingen", "Ludwigsburg", "Heidelberg", "stuttgart"],
            }

class ZDLVectorModel:

    def __init__(self, read_pickle: bool, classifier, locale_type: str = "all", path: str = 'data') -> None:

        """
        This class builds a regiolect prediction model, using ZDL Regionalkorpus resources
        using which we can embed tokens in 6-dimensional vectors that represent their usage in different areas of Germany
        :param read_pickle: whether to read an already an existing pickle file with training data or not
        :param classifier: the classifier to use
        :param locale_type: "all", "EAST_WEST", or "NORTH_SOUTH"
        """

        if not locale_type in ["all", "EAST_WEST", "NORTH_SOUTH"]:
            raise ValueError('locale_type should be one of "all", "EAST_WEST", or "NORTH_SOUTH".')
        
        if not isinstance(read_pickle, bool):
            raise ValueError('read_pickle has to be True or False')
        
        self.path = path
        self.locale_type = locale_type
        # dictionary linking cities from subreddits to the areal they belong to
        self.areal_dict = AREAL_DICT
        
        with open("vectors/zdl_vector_dict.json", "r") as f:
            self.vector_dict = json.load(f)

        self.classifier = classifier
        if read_pickle:
            data = pd.read_pickle("vectors/zdl_vector_matrix.pickle")
        else:
            data = self.__train()
        self.training_data = self.__create_training_set_from_data(data)
        self.model = self.__build_model()

    def __create_training_set_from_data(self, data: pd.DataFrame):
        """ adds padding to vectors and makes format readable for classifier """
        maxVal = self.__find_longest_vector(data)
        data["padded_vectors"] = [self.__zero_pad_vector(vector, maxVal) for vector in data["vector"].tolist()]
        data["simple_locale"] = [self.__simplify_locale(locale) for locale in data["LOCALE"].tolist()]
        data["LOCALE_NUM"] = [self.__locale_to_num(locale) for locale in data["LOCALE"].tolist()]

        return data
    
    def __build_model(self):

        X = self.training_data["padded_vectors"].tolist()

        if self.locale_type == "all":
            y = self.training_data["LOCALE_NUM"]
        else:
            y = self.training_data["simple_locale"]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y)
        
        model = self.classifier
        model.fit(self.X_train, self.y_train)

        return model

    @staticmethod
    def _json_to_vector(response: dict) -> np.array:
        """ turns the ZDL response into a 6d vector """
        ppm_values = [item["ppm"] for item in response]
        return np.array(ppm_values)

    @staticmethod
    def _vectorize_sample(text: str, vectionary: dict) -> np.array:
        """ tokenizes and vectorizes a text sample using ZDL regionalkorpus """
        vectors = []
        text = text.replace("\n", "") 
        punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
        for x in text.lower(): 
            if x in punctuations: 
                text = text.replace(x, "") 
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        tokens = tokenize(text) # tokenize the sample
        for token in tokens:
            if token in stopwords_:
                continue
            # if token has not been called yet, do api call
            if token not in vectionary:
                try:
                    r = zdl_request(token)
                except (requests.exceptions.JSONDecodeError, ValueError):
                    continue    # some words return no result
                vector = ZDLVectorModel._json_to_vector(r)
                vectionary[token] = vector.tolist()   # save the vector as a list (for json)
            else:
                # else get vector from dict
                vector = vectionary[token]
                #tqdm.write('got vector from dict')
            #tqdm.write(str(vector))
            vectors.append(vector)
        # keep only complete vectors
        return np.array([v for v in vectors if len(v) == 6])

    def __find_longest_vector(self, data: pd.DataFrame) -> int:
        # for zero padding we need to know the length to which we want to pad
        maxVal = 0
        for vector in data["vector"].tolist():
            val = len(vector)
            if val > maxVal:
                maxVal = val
        return maxVal

    def __check_text_is_german(self, text: str) -> bool:
        """ return true if text is german else false """
        DetectorFactory.seed = 0
        # we skip very short texts or posts removed from reddit
        if text in ("[removed]", "[deleted]")  or len(text) < 20 or len(text.split()) < 3:
            return False
        try:
            lang = detect(text)
        except lang_detect_exception.LangDetectException:
            return False
        
        tqdm.write(str(lang))
        return lang == "de"

    def __zero_pad_vector(self, vector: np.array, maxval: int) -> list:
        """ pad vector to length of longest vector in dataset """

        zero_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # empty vector
        vector_length = len(vector)  # length of target vector
        diff = maxval - vector_length   # amount of padding needed

        vector = np.append(vector, [zero_vec] * diff)

        return vector #.reshape(-1,1)

    def __simplify_locale(self, locale: str) -> str:

        if self.locale_type == "NORTH_SOUTH":
            locales = {
                "DE-NORTH": ["DE-NORTH-WEST", "DE-NORTH-EAST"],
                "DE-SOUTH": ["DE-SOUTH-WEST", "DE-SOUTH-EAST", "DE-MIDDLE-WEST", "DE-MIDDLE-EAST"]
                }

        elif self.locale_type == "EAST_WEST":
            locales = {
                "DE-EAST": ["DE-NORTH-EAST", "DE-MIDDLE-EAST", "DE-SOUTH-EAST"],
                "DE-WEST": ["DE-NORTH-WEST", "DE-MIDDLE-WEST", "DE-SOUTH-WEST"]
            }
        else:
            return locale
        
        for key in locales.keys():
            if locale in locales[key]:
                return key
        
    def __locale_to_num(self, locale: str) -> int:
        locs = {
            "DE-NORTH-EAST": 1,
            "DE-NORTH-WEST": 2,
            "DE-MIDDLE-EAST": 3,
            "DE-MIDDLE-WEST": 4,
            "DE-SOUTH-EAST": 5,
            "DE-SOUTH-WEST": 6
        }
        return locs[locale]

    def __train(self):
        directory_in_str = f"{self.path}/reddit/locales"
        directory = os.fsencode(directory_in_str)
        dataframe_list = []   
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".json"): 
                with open(f"{directory_in_str}/{filename}", "r") as f:
                    file_data = json.load(f)
                    for key in self.areal_dict.keys():
                        if filename.split(".")[0] in self.areal_dict[key]:
                            current_data = pd.DataFrame()
                            current_data["texts"] = list(set([item["selftext"] for item in file_data["data"] if self.__check_text_is_german(item["selftext"])]))
                            current_data["LOCALE"] = key
                            dataframe_list.append(current_data)

        training_data = pd.concat(dataframe_list)
        training_data["vector"] = [self._vectorize_sample(text) for text in tqdm(training_data["texts"].tolist())]
        print(training_data.head())
        training_data.reset_index(inplace=True, drop=True)
        training_data.to_pickle("vectors/zdl_vector_matrix.pickle")

        with open("vectors/zdl_vector_dict.json", "w") as f:
            json.dump(self.vector_dict, f)

        return training_data
    
    def evaluate(self):

        y_pred = self.model.predict(self.X_test)
        #y_pred = np.round(y_pred)

        target_names = [
                "DE-NORTH-EAST", 
                "DE-NORTH-WEST", 
                "DE-MIDDLE-EAST", 
                "DE-MIDDLE-WEST",
                "DE-SOUTH-EAST",
                "DE-SOUTH-WEST"]
        
        report = classification_report(
            self.y_test, 
            y_pred, 
            output_dict=True,
            target_names=target_names if self.locale_type == "all" else None)

        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Greens")
        plt.show()


class ZDLVectorMatrix:

    def __init__(self, source: DataCorpus = None, path: str = None) -> None:
        """
        a slightly buggy class which generates a vector matrix for a DataCorpus
        :param source: DataCorpus object
        :param path: if vectors have been stored as a json, they can be loaded
        """
        self.data: DataCorpus = source
        with open('vectors/zdl_vector_dict.json', 'r', encoding='utf-8') as f:
            self.vectionary: dict = json.load(f)
        if path:
            with open('data/vectors/ZDLCorpus_data.json', 'r') as f:
                self.vectors = json.load(f)
        else:
            self.vectors = self.__vectorize_data()
            
    def __call_vectorizer(self, sample: str) -> np.array:
        """
        use vectorize  method from other class, passing the 
        vector dict from this class to it
        """
        return ZDLVectorModel._vectorize_sample(sample, self.vectionary)
    
    def __vectorize_data(self):
        print("BUILDING ZDL MATRIX")
        matrix = {}
        for obj in tqdm(self.data.corpus):
            ID = obj.content['id']
            V = self.__call_vectorizer(obj.text)
            matrix[ID] = V

        return matrix
    
    def save_to_json(self):
        with open('data/vectors/ZDLCorpus_data.json', 'w') as f:
            json.dump(self.vectors, f)


if __name__ == "__main__":

    model = ZDLVectorModel(
        read_pickle=True, 
        classifier=MLPClassifier(),
        locale_type='all')
    model.evaluate()

    exit()

    estimator = model.estimators_[0]
    export_graphviz(estimator, out_file='tree.dot', 
                feature_names = ["Vectors"] * 1560,
                class_names = target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

    from subprocess import call
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    sample = """Moin in die Runde,

ich hab auf meinem Balkon beobachtet, wie ab und zu Bienen in einen kleinen Spalt fliegen. Hab mir nichts dabei gedacht und es einfach mal zu geklebt. Das war im nach hinein ein fataler Fehler. Denn jetzt fliegen da etwa 30-40 Bienen herum, was vermuten lässt, dass da ein ganzes Nest hinter der nebenstehenden Holzverkleidung ist. Was kann man dazu in Bremen tun? Kammerjäger ist eigentlich keine Option, da die ja unter Naturschutz stehen und ich sie eher umsiedeln möchte. Kennt ihr vielleicht jemanden oder eine Idee was ich machen kann?"""
    sample_vector = _vectorize_sample(sample)
    sample_vector = __zero_pad_vector(sample_vector, maxVal)
    sample_vector = sample_vector.reshape(1, -1)
    print(model.predict_proba(sample_vector))

