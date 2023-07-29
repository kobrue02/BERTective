import json
import re
import pandas as pd

from pprint import pprint
from tqdm import tqdm

import numpy as np
#import tensorflow as tf
#import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

#tf.saved_model.LoadOptions(experimental_io_device='/job:localhost')
#import tensorflow_text

#def train_model(data: dict, path: str):
#    use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
#
#    use_vectors = np.array([use_embed([str(post['text'])]).numpy() for post in tqdm(data['texts'])])
#    use_vectors = np.squeeze(use_vectors)
#
#    with open(f"vectors/{path}.npy", "wb") as f:
#        np.save(f, use_vectors)

def train_model_doc2vec(data: dict, path):
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(tqdm(data['texts']))]
    model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=8)

    vectors = []
    try:
        for vector in model.docvecs:
            vectors.append(vector)
    except KeyError:
        pass

    use_vectors = np.array(vectors)
    use_vectors = np.squeeze(use_vectors)

    with open(f"vectors/{path}_gensim.npy", "wb") as f:
        np.save(f, use_vectors)

    
def vector_df(data: dict, path: str) -> pd.DataFrame:

    df = pd.DataFrame()

    with open(f"vectors/{path}.npy", "rb") as f:
        use_vectors = [_ for _ in np.load(f)]

    df['vectors'] = use_vectors
    df['author_name'] = [item['author_name'] for item in data['texts']]
    
    return df


if __name__ == "__main__":

    with open('data.json', 'r') as f:
        data = json.load(f)

    data['texts'] = data['texts'][:200]

    train_model_doc2vec(data, 'gutenberg')
    #train_model(data, 'gutenberg')
    model_dataset = vector_df(data, 'gutenberg_gensim')
    print(model_dataset.head())

    model = SVC()
    scaler = MinMaxScaler()
    X, y = model_dataset['vectors'].tolist(), model_dataset['author_name'].tolist()

    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred))
