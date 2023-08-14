import pandas as pd
import numpy as np

import os
import re
import json
import requests
from tqdm import tqdm
from scraping_tools.zdl_regio_client import zdl_request, tokenize
from langdetect import detect, DetectorFactory, lang_detect_exception
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz

from german_stopwords import stopwords_

import tensorflow as tf
import seaborn as sns

areal_dict = {
    "DE-NORTH-EAST": ["Rostock", "Berlin","luebeck", "Potsdam"],
    "DE-NORTH-WEST": ["bremen", "Hamburg","Hannover", "bielefeld", "Dortmund", "kiel", "Paderborn"],
    "DE-MIDDLE-EAST": ["Leipzig", "dresden", "HalleSaale", "erfurt", "jena"],
    "DE-MIDDLE-WEST": ["frankfurt", "duesseldorf", "cologne","Bonn", "Mainz", "Wiesbaden", "kaiserslautern", "Mannheim", "Saarland"],
    "DE-SOUTH-EAST": ["Munich", "Nurnberg", "Regensburg", "Würzburg", "bayreuth", "bavaria"],
    "DE-SOUTH-WEST": ["stuttgart", "augsburg", "freiburg", "karlsruhe", "Ulm", "Tuebingen", "Ludwigsburg", "Heidelberg", "stuttgart"],
}

#vector_dict = {}
with open("vectors/zdl_vector_dict.json", "r") as f:
    vector_dict = json.load(f)

def json_to_vector(response: dict) -> np.array:
    """ turns the ZDL response into a 6d vector """
    ppm_values = [item["ppm"] for item in response]
    return np.array(ppm_values)

def vectorize_sample(text: str) -> np.array:
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
        if token not in vector_dict:
            try:
                r = zdl_request(token)
            except (requests.exceptions.JSONDecodeError, ValueError):
                continue    # some words return no result
            vector = json_to_vector(r)
            vector_dict[token] = vector.tolist()   # save the vector as a list (for json)
        else:
            # else get vector from dict
            vector = vector_dict[token]
            tqdm.write('got vector from dict')
        tqdm.write(str(vector))
        vectors.append(vector)
    # keep only complete vectors
    return np.array([v for v in vectors if len(v) == 6])

def find_longest_vector(data: pd.DataFrame) -> int:
    # for zero padding we need to know the length to which we want to pad
    maxVal = 0
    for vector in data["vector"].tolist():
        val = len(vector)
        if val > maxVal:
            maxVal = val
    return maxVal

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
    tqdm.write(str(lang))
    return lang == "de"

def zero_pad_vector(vector: np.array, maxval: int) -> list:
    """ pad vector to length of longest vector in dataset """

    zero_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # empty vector
    vector_length = len(vector)  # length of target vector
    diff = maxval - vector_length   # amount of padding needed

    vector = np.append(vector, [zero_vec] * diff)

    return vector #.reshape(-1,1)

def simplify_locale(locale: str) -> str:
    locales = {
        "DE-NORTH": ["DE-NORTH-WEST", "DE-NORTH-EAST"],
        "DE-SOUTH": ["DE-SOUTH-WEST", "DE-SOUTH-EAST", "DE-MIDDLE-WEST", "DE-MIDDLE-EAST"]
        }
    
    locales_2 = {
        "DE-EAST": ["DE-NORTH-EAST", "DE-MIDDLE-EAST", "DE-SOUTH-EAST"],
        "DE-WEST": ["DE-NORTH-WEST", "DE-MIDDLE-WEST", "DE-SOUTH-WEST"]
    }

    for key in locales_2.keys():
        if locale in locales_2[key]:
            return key
        
def locale_to_num(locale: str) -> int:
    locs = {
        "DE-NORTH-EAST": 1,
        "DE-NORTH-WEST": 2,
        "DE-MIDDLE-EAST": 3,
        "DE-MIDDLE-WEST": 4,
        "DE-SOUTH-EAST": 5,
        "DE-SOUTH-WEST": 6
     }
    return locs[locale]

def train():
    directory_in_str = "data/reddit/locales"
    directory = os.fsencode(directory_in_str)
    dataframe_list = []   
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".json"): 
            with open(f"{directory_in_str}/{filename}", "r") as f:
                file_data = json.load(f)
                for key in areal_dict.keys():
                    if filename.split(".")[0] in areal_dict[key]:
                        current_data = pd.DataFrame()
                        current_data["texts"] = list(set([item["selftext"] for item in file_data["data"] if check_text_is_german(item["selftext"])]))
                        current_data["LOCALE"] = key
                        dataframe_list.append(current_data)

    training_data = pd.concat(dataframe_list)
    training_data["vector"] = [vectorize_sample(text) for text in tqdm(training_data["texts"].tolist())]
    print(training_data.head())
    training_data.reset_index(inplace=True, drop=True)
    training_data.to_pickle("vectors/zdl_vector_matrix.pickle")
    with open("vectors/zdl_vector_dict.json", "w") as f:
        json.dump(vector_dict, f)

    return training_data

def call_keras_model(X_train, X_test, y_train, y_test):
    from models.keras_cnn_implementation import model, batch_size, epochs
    X_train = tf.stack(X_train)
    X_test = tf.stack(X_test)
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test,y_test))
    print(model.summary())
    y_pred = model.predict(X_test)
    y_pred = np.round(y_pred)
    report = classification_report(y_test, y_pred)
    print(report)

if __name__ == "__main__":

    training_data = train()

    #training_data = pd.read_pickle("vectors/zdl_vector_matrix.pickle")

    maxVal = find_longest_vector(training_data)

    training_data["padded_vectors"] = [zero_pad_vector(vector, maxVal) for vector in training_data["vector"].tolist()]
    training_data["simple_locale"] = [simplify_locale(locale) for locale in training_data["LOCALE"].tolist()]
    training_data["LOCALE_NUM"] = [locale_to_num(locale) for locale in training_data["LOCALE"].tolist()]

    print(training_data.head())
    X, y = training_data["padded_vectors"].tolist(), training_data["LOCALE_NUM"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y)
    
    model = MLPClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #y_pred = np.round(y_pred)

    target_names = [
            "DE-NORTH-EAST", 
            "DE-NORTH-WEST", 
            "DE-MIDDLE-EAST", 
            "DE-MIDDLE-WEST",
            "DE-SOUTH-EAST",
            "DE-SOUTH-WEST"]
    
    report = classification_report(
        y_test, 
        y_pred, 
        output_dict=True,
        target_names=target_names)
    print(report)

    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Greens")
    plt.show()
    
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
    sample_vector = vectorize_sample(sample)
    sample_vector = zero_pad_vector(sample_vector, maxVal)
    sample_vector = sample_vector.reshape(1, -1)
    print(model.predict_proba(sample_vector))

