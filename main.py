from corpus import DataCorpus, DataObject
from crawl_all_datasets import download_data
from models.zdl_vector_model import AREAL_DICT
from models.wiktionary_matrix import WiktionaryModel
from tqdm import tqdm
from langdetect import detect, DetectorFactory, lang_detect_exception
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import argparse
import json
import nltk
import os
import pandas as pd
import random
import time

def __reddit_to_datacorpus(path: str, corpus: DataCorpus):
    directory_in_str = f"{path}/reddit/locales"
    directory = os.fsencode(directory_in_str)
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".json"): 
            with open(f"{directory_in_str}/{filename}", "r") as f:
                file_data = json.load(f)
                for key in AREAL_DICT.keys():
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
    achse = pd.read_parquet(f'{path}/achse/achse_des_guten_annotated_items.parquet')

    for item in zip(achse.content, achse.age, achse.sex):
        obj = DataObject(
            text = item[0],
            author_age=item[1],
            author_gender=item[2],
            source="ACHGUT"
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
    data.read_avro(f'{PATH}/corpus.avro')
    length = len(data)
    print(length)
    #corpus = __reddit_to_datacorpus(PATH, data)
    #length = len(data)
    #corpus = __achgut_to_datacorpus(PATH, data)
    #print(length)
    #data.save_to_avro(f"{PATH}/corpus.avro")
    #time.sleep(2)
    
    wiktionary_matrix = WiktionaryModel(data)
    #wiktionary_matrix = WiktionaryModel('data/wiktionary/wiktionary.parquet')
    #print(wiktionary_matrix.vectors)

    ids_ = []

    for item in data.corpus:
        if item.source == "ACHGUT":
            ids_.append(item.content['id'])
        else:
            break

    y = [float(data[id].author_age) for id in ids_]
    X = [wiktionary_matrix[id] for id in ids_]

    X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)

    model = SVR()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    offset = 0
    for i in zip(y_test, y_pred):
        print(i)
        offset += (abs(i[0] - i[1]))
    print('-'*64)
    print(offset / len(y_test))

    #print(classification_report(y_test, y_pred))
    
    #wiktionary_df = pd.read_parquet('data/wiktionary/wiktionary.parquet')
    #print(wiktionary_df.head())


    
    wiktionary_matrix.df_matrix.to_parquet('data/wiktionary/wiktionary.parquet')
    exit()
    download_data(['achse', 'ortho', 'reddit_locales'], "test")

    data = DataCorpus()


    print(data.as_dataframe().head())
    data.save_to_avro(f"{PATH}/corpus.avro")
