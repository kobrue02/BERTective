import json
import nltk
import os
import pandas as pd
import pickle
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

        """
        Generate wiktionary feature vectors based on the word lists that have been crawled.
        Vectors are stored with the ID of the DataObject they represent.
        :param path: if matrix has been generated previously, it can be loaded
        :param source: DataCorpus object
        """

        # if path is passed, try to load parquet file
        if path:
            self.df_matrix: pd.DataFrame = self.__read_parquet(path)
            self.vectors = self.__vectors_from_df()
        
        # otherwise build matrix from scratch using wiktionary file
        else:
            self.data: DataCorpus = source
            with open('data/wiktionary/wiktionary.json', 'r', encoding='utf-8') as f:
                self.wiktionary: dict = json.load(f)
            self.vectors = {}
            self.__wiktionary_matrix = self.__build_matrix()
            self.df_matrix: pd.DataFrame = self.__to_df()
        

    def __build_matrix(self):
        """
        for each DataObject in the corpus, the text is tokenized
        and matched with the wiktionary lexicon entries.
        the matches are stored in arrays.
        
        :returns: a dictionary which has the DataObject IDs as keys
        and the generated vectors as values.
        """
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
        """
        turns a wiktionary matrix into a pandas dataframe.
    	this dataframe can be stored locally and re-read.
        :returns: dataframe representing the wiktionary matrix.
        """

        df = pd.DataFrame()

        df['DataObject_ID'] = list(self.__wiktionary_matrix.keys())

        for KEY in list(self.wiktionary.keys()):
            df[KEY] = [obj[KEY] for obj in self.__wiktionary_matrix.values()]

        return df
    
    def __to_vector(self, dist: dict) -> np.array:
        """
        turns a dictionary containing wiktionary matches into a vector. 

        :returns: numpy array with n dimensions where n is the amount of wiktionary lexicon entries. """
        vector = []

        for key in list(dist.keys()):
            v = dist[key]
            vector.append(v)

        return np.array(vector)

    def __read_parquet(self, path: str):
        """ 
        read the pandas dataframe representation of a
        wiktionary matrix which was stored locally in 
        a parquet file.
        :param path:  the path to the parquet file.
        """
        df = pd.read_parquet(path)
        return df
    
    def __vectors_from_df(self):
        """
        turns the columns of a dataframe into n-dimensional vectors where
        n is the amount of columns (ID not included).

        :returns: dictionary which has DataObject ID as keys and vectors as values.
        """

        # if vectors have been generated before
        vectors_exist = os.path.isfile('vectors/wiktionary.pickle')
        if vectors_exist:
            print("LOADING WIKTIONARY VECTORS")
            with open('vectors/wiktionary.pickle', 'rb') as f:
                vectors = pickle.load(f)
                return vectors

        print("No vector database found, building new one.")
        # if file does not exist, build from scratch
        vectors = {}
        columns = self.df_matrix.columns.values.tolist()[1:]
        for i in tqdm(range(len(self.df_matrix.index))):
            vector = np.array([self.df_matrix[col].tolist()[i] for col in columns])
            vectors[i] = vector
        
        with open('vectors/wiktionary.pickle', 'wb') as f:
            pickle.dump(vectors, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('Saved vector database to file.')
        return vectors
    
    def __getitem__(self, i):
        return self.vectors[i]
            


