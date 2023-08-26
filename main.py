from corpus import DataCorpus, DataObject
from crawl_all_datasets import download_data
from models.zdl_vector_model import AREAL_DICT
from models.wiktionary_matrix import WiktionaryModel
from models.keras_cnn_implementation import build_cnn_model

from tqdm import tqdm
from langdetect import detect, DetectorFactory, lang_detect_exception
from keras.backend import clear_session


from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, BayesianRidge, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import argparse
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import random
import time

def __reddit_locales_to_datacorpus(path: str, corpus: DataCorpus):
    """ finds reddit files and adds them to a DataCorpus """
    directory_in_str = f"{path}/reddit/locales"
    directory = os.fsencode(directory_in_str)

    # iterating over all files in the directory
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".json"): 
            # json files contain reddit data
            with open(f"{directory_in_str}/{filename}", "r") as f:
                file_data = json.load(f)
                for key in AREAL_DICT.keys():
                    # find the areal to which the city belongs
                    if filename.split(".")[0] in AREAL_DICT[key]:
                        for item in file_data["data"]:
                            text = item['selftext']
                            if check_text_is_german(text):
                                obj = DataObject(
                                        text = text,
                                        author_regiolect=key,
                                        source="REDDIT"
                                    )
                                corpus.add_item(obj)
    return corpus
    
def __achgut_to_datacorpus(path: str, corpus: DataCorpus):
    """ reads achgut blog posts and adds to DataCorpus """
    achse = pd.read_parquet(f'{path}/achse/achse_des_guten_annotated_items.parquet')

    for item in zip(achse.content, achse.age, achse.sex, achse.education, achse.regiolect):
        obj = DataObject(
            text = item[0],
            author_age=item[1],
            author_gender=item[2],
            source="ACHGUT",
            author_education=item[3],
            author_regiolect=item[4])

        corpus.add_item(obj)

    return corpus

def __reddit_to_datacorpus(path: str, corpus: DataCorpus):
    """ 
    reads reddit posts and adds to DataCorpus 
    :param path: path to parquest file with annotated posts
    :param corpus: DataCorpus object to append data to
    """
    reddit = pd.read_parquet(f'{path}/reddit/annotated_posts.parquet')

    for item in zip(reddit.content, reddit.age, reddit.sex, reddit.regiolect):
        try:
            regiolect = item[3]
        except IndexError:
            regiolect = None
        obj = DataObject(
            text = item[0],
            author_age=item[1],
            author_gender=item[2],
            author_regiolect=regiolect,
            source="REDDIT"
        )

        corpus.add_item(obj)

    return corpus
                                

def check_text_is_german(text: str) -> bool:
    """ return true if text is german else false """
    DetectorFactory.seed = 0
    # we skip very short texts or posts removed from reddit
    if text in ("[removed]", "[deleted]")  or len(text) < 20 or len(text.split()) < 3:
        return False
    try:
        lang = detect(text)
    except lang_detect_exception.LangDetectException:
        return False
        
    return lang == "de"

def __build_corpus(data: DataCorpus, PATH: str) -> DataCorpus:
    print("loading reddit data from locales")
    data = __reddit_locales_to_datacorpus(PATH, data)
    print("loading achgut data")
    data = __achgut_to_datacorpus(PATH, data)
    print("loading reddit data from miscellaneous")
    data = __reddit_to_datacorpus(PATH, data)
    return data



if __name__ == "__main__":

    PATH = "test"

    os.makedirs(PATH, exist_ok=True)
    os.makedirs(f'{PATH}/achse', exist_ok=True)
    os.makedirs(f'{PATH}/annotation', exist_ok=True)
    os.makedirs(f'{PATH}/reddit', exist_ok=True)
    os.makedirs(f'{PATH}/reddit/locales', exist_ok=True)


    with open('data/wiktionary/wiktionary.json', 'r', encoding='utf-8') as f:
        wiktionary: dict = json.load(f)

    data = DataCorpus()

    # if we want to build corpus
    #data = __build_corpus(data, PATH)
    #data.save_to_avro(f"{PATH}/corpus.avro")

    # read existing corpus
    data.read_avro(f'{PATH}/corpus.avro')
    

    #wiktionary_matrix = WiktionaryModel(source=data)
    wiktionary_matrix = WiktionaryModel('data/wiktionary/wiktionary.parquet')
    #wiktionary_matrix.df_matrix.to_parquet('data/wiktionary/wiktionary.parquet')
    

    ids_ = []

    for item in data.corpus:
        if item.source in ("REDDIT"):
            if item.author_regiolect not in ("N/A", "NONE", "", None):
                ids_.append(item.content['id'])
        else:
            continue

    y = [data[id].author_regiolect for id in ids_]
    X = [wiktionary_matrix[id] for id in ids_]

    X = np.asarray(X).reshape(-1, 27, 1)
    y = np.asarray(y)
    
    for item in list(set(y)):
        print(f"{item.upper()}: {list(y).count(item)}")

    n_inputs, n_outputs = 27, y.shape[0]

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(list(set(y_train)))

    def to_num(L: list) -> list:

        a = {
            "DE-MIDDLE-EAST": 1.0,
            "DE-MIDDLE-WEST": 2.0,
            "DE-NORTH-EAST": 3.0,
            "DE-NORTH-WEST": 4.0,
            "DE-SOUTH-EAST": 5.0,
            "DE-SOUTH-WEST": 6.0
            }
        
        b = {"finished_highschool": 1.0,
             "has_phd": 2.0,
             "has_apprentice": 3.0,
             "has_master": 4.0}

        c = {"female": 1.0,
             "male": 2.0}
        
        if L[0] in list(a.keys()):
            return [a[item] for item in L]
        elif L[0] in list(b.keys()):
            return [b[item] for item in L]
        else: 
            return [c[item] for item in L]
    
    y_train = np.asarray(to_num(y_train))
    y_test = np.asarray(to_num(y_test))
    

    clear_session()
    model = build_cnn_model(n_inputs, n_outputs)
    print(model.summary())

    history = model.fit(X_train, y_train, 
                        epochs=256, 
                        verbose=True, 
                        validation_data=(X_test, y_test), 
                        batch_size=128,
                        use_multiprocessing=True,
                        workers=6)

    loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
    print("Testing Accuracy:  {:.4f}".format(accuracy))
    exit()

    #model = MLPClassifier()
    # for regression
    #model = KNeighborsRegressor(
    #    weights='distance', 
    #    n_neighbors=14)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))

    exit()
    offset = 0
    for j in zip(y_test, y_pred):
        offset += (abs(j[0] - int(j[1])))
        avg = offset / len(y_test)
    print(avg)
    
    #plt.plot(y_test)
    #plt.plot(y_pred, 'o')

    #plt.show()

    #print(classification_report(y_test, y_pred))
    
    #wiktionary_df = pd.read_parquet('data/wiktionary/wiktionary.parquet')
    #print(wiktionary_df.head())


    
    wiktionary_matrix.df_matrix.to_parquet('data/wiktionary/wiktionary.parquet')
    exit()
    download_data(['achse', 'ortho', 'reddit_locales'], "test")

    data = DataCorpus()


    print(data.as_dataframe().head())
    data.save_to_avro(f"{PATH}/corpus.avro")
