import json
import nltk
import os
import pandas as pd
import numpy as np
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
# to import from parent directionary
from corpus import DataCorpus, DataObject
from tqdm import tqdm


class WiktionaryModel:

    def __init__(self, path: str = None, source: DataCorpus = None) -> None:

        if path:
            self.df_matrix: pd.DataFrame = self.__read_parquet(path)
            self.vectors = self.__vectors_from_df()
        
        else:
            self.data: DataCorpus = source
            with open('data/wiktionary/wiktionary.json', 'r', encoding='utf-8') as f:
                self.wiktionary: dict = json.load(f)
            self.vectors = {}
            self.__wiktionary_matrix = self.__build_matrix()
            self.df_matrix: pd.DataFrame = self.__to_df()
        

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
            self.vectors[ID] = self.__to_vector(dist)
        return matrix
    
    def __to_df(self):

        df = pd.DataFrame()

        df['DataObject_ID'] = list(self.__wiktionary_matrix.keys())

        for KEY in list(self.wiktionary.keys()):
            df[KEY] = [obj[KEY] for obj in self.__wiktionary_matrix.values()]

        return df
    
    def __to_vector(self, dist: dict) -> np.array:
        vector = []

        for key in list(dist.keys()):
            v = dist[key]
            vector.append(v)

        return np.array(vector)

    def __read_parquet(self, path: str):
        df = pd.read_parquet(path)
        return df
    
    def __vectors_from_df(self):
        vectors = {}
        columns = self.df_matrix.columns.values.tolist()[1:]
        for i in tqdm(range(len(self.df_matrix.index))):
            vector = []
            for col in columns:
                vector.append(self.df_matrix[col].tolist()[i])
            vectors[i] = vector
        
        return vectors
    
    def __getitem__(self, i):
        return self.vectors[i]
            


