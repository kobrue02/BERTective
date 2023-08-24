import json
import nltk
import os
import pandas as pd
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# to import from parent directionary
from corpus import DataCorpus, DataObject
from tqdm import tqdm


class WiktionaryModel:

    def __init__(self, source: DataCorpus) -> None:

        self.data: DataCorpus = source
        with open('data/wiktionary/wiktionary.json', 'r', encoding='utf-8') as f:
            self.wiktionary: dict = json.load(f)
        self.wiktionary_matrix = self.__build_matrix()
        self.matrix_as_dataframe = self.__to_df()

    def __build_matrix(self):
        print("BUILDING WIKTIONARY MATRIX")
        matrix = {}
        for obj in tqdm(self.data.corpus):
            ID = obj.content['id']

            tokens = nltk.word_tokenize(obj.text.lower())
            dist = {k: 0 for k in list(self.wiktionary.keys())}

            for key in list(self.wiktionary.keys()):
                for word in self.wiktionary[key]:
                    if word.lower() in tokens:
                        dist[key] += 1

            matrix[ID] = dist
        return matrix
    
    def __to_df(self):

        df = pd.DataFrame()

        df['DataObject_ID'] = list(self.wiktionary_matrix.keys())

        for KEY in list(self.wiktionary.keys()):
            df[KEY] = [obj[KEY] for obj in self.wiktionary_matrix.values()]

        return df