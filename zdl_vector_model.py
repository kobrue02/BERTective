import pandas as pd
import numpy as np

import os

import json
import requests
from tqdm import tqdm
from scraping_tools.zdl_regio_client import zdl_request, tokenize
from langdetect import detect, DetectorFactory

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle


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

def vectorize_sample(text: str):
    """ tokenizes and vectorizes a text sample using ZDL regionalkorpus """
    vectors = []
    tokens = tokenize(text) # tokenize the sample
    for token in tokens:
        # if token has not been called yet, do api call
        if token not in vector_dict:
            try:
                r = zdl_request(token)
            except (requests.exceptions.JSONDecodeError, ValueError):
                continue
            vector = json_to_vector(r)
            vector_dict[token] = vector.tolist()
        else:
            # else get vector from dict
            vector = vector_dict[token]
            tqdm.write('got vector from dict')
        tqdm.write(str(vector))
        vectors.append(vector)
    return np.array([v for v in vectors if len(v) == 6])

def find_longest_vector(data: pd.DataFrame) -> int:
    maxVal = 0
    for vector in data["vector"].tolist():
        val = len(vector)
        if val > maxVal:
            maxVal = val
    return maxVal

def check_text_is_german(text: str) -> bool:
    DetectorFactory.seed = 0
    if text in ("[removed]", "[deleted]")  or len(text) < 20 or len(text.split()) < 3:
        return False
    lang = detect(text)
    tqdm.write(str(lang))
    return lang == "de"

def zero_pad_vector(vector: np.array, maxval: int) -> list:

    zero_vec = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # empty vector
    vector_length = len(vector)  # length of target vector
    diff = maxval - vector_length   # amount of padding needed

    vector = np.append(vector, [zero_vec] * diff)

    return vector

def simplify_locale(locale: str) -> str:
    locales = {
        "DE-NORTH": ["DE-NORTH-WEST", "DE-NORTH-EAST"],
        "DE-MIDDLE": ["DE-MIDDLE-WEST", "DE-MIDDLE-EAST"],
        "DE-SOUTH": ["DE-SOUTH-WEST", "DE-SOUTH-EAST"]
        }
    
    locales_2 = {
        "DE-EAST": ["DE-NORTH-EAST", "DE-MIDDLE-EAST", "DE-SOUTH-EAST"],
        "DE-WEST": ["DE-NORTH-WEST", "DE-MIDDLE-WEST", "DE-SOUTH-WEST"]
    }

    for key in locales.keys():
        if locale in locales[key]:
            return key

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
                        current_data["texts"] = [item["selftext"] for item in file_data["data"] if check_text_is_german(item["selftext"])]
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

if __name__ == "__main__":

    #training_data = train()

    training_data = pd.read_pickle("vectors/zdl_vector_matrix.pickle")

    maxVal = find_longest_vector(training_data)

    training_data["padded_vectors"] = [zero_pad_vector(vector, maxVal) for vector in training_data["vector"].tolist()]
    training_data["simple_locale"] = [simplify_locale(locale) for locale in training_data["LOCALE"].tolist()]

    model = MLPClassifier()
    X, y = training_data["padded_vectors"].tolist(), training_data["LOCALE"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))



