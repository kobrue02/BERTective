import json
import re
import pandas as pd

from pprint import pprint
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import numpy as np

tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
import tensorflow_text

def train_model(data: dict, path: str):
    use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

    use_vectors = np.array([use_embed([str(post['selftext'])]).numpy() for post in tqdm(data['data'])])
    use_vectors = np.squeeze(use_vectors)

    with open(f"vectors/{path}.npy", "wb") as f:
        np.save(f, use_vectors)


def vector_df(label: str) -> pd.DataFrame:

    df = pd.DataFrame()

    with open(f"vectors/{label}.npy", "rb") as f:
        use_vectors = [_ for _ in np.load(f)]

    df['vectors'] = use_vectors
    df['label'] = len(use_vectors)*[label]
    
    return df


if __name__ == "__main__":

    with open('data/reddit/profession/bundeswehr.json', "r") as f:
        bundeswehr_data = json.load(f)

    with open('data/reddit/profession/feuerwehr.json', "r") as f:
        feuerwehr_data = json.load(f)

    with open('data/reddit/profession/oeffentlicherdienst.json', "r") as f:
        od_data = json.load(f)

    with open('data/reddit/education/Azubis.json', "r") as f:
        azubi_data = json.load(f)

    with open('data/reddit/education/Studium.json', "r") as f:
        studi_data = json.load(f)

    with open('data/reddit/dating/relationship_advice.json', "r") as f:
        dating_data = json.load(f)

    #train_model(bundeswehr_data, 'bundeswehr')
    #train_model(feuerwehr_data, 'feuerwehr')
    #train_model(od_data, 'oeffentlicherdienst')
    #train_model(studi_data, 'student')
    #train_model(azubi_data, 'azubi')
    #train_model(dating_data, 'dating')

    bundeswehr_df = vector_df('bundeswehr')
    feuerwehr_df = vector_df('feuerwehr')
    od_df = vector_df('oeffentlicherdienst')
    azubi_df = vector_df('azubi')
    studi_df = vector_df('student')
    dating_df = vector_df('dating')

    combined_df = pd.concat([bundeswehr_df, feuerwehr_df, od_df, azubi_df, studi_df, dating_df])

    model = LogisticRegression()
    scaler = MinMaxScaler()
    X, y = combined_df['vectors'].tolist(), combined_df['label'].tolist()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))