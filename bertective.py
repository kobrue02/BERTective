import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForMaskedLM

from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.utils import shuffle

import argparse

import umap
import umap.plot

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

from annotate_csv import annotate_to_dataframe
from sklearn.model_selection import train_test_split

# achse des guten
file_1 = pd.read_excel("data/achse/achse_des_guten_annotated_10000_items.xlsx")#.drop(["age"], axis=1)
file_2 = pd.read_excel("data/achse/achse_des_guten_annotated_20000_items.xlsx")#.drop(["age"], axis=1)
file_3 = pd.read_excel("data/achse/achse_des_guten_annotated_30000_items.xlsx")#.drop(["age"], axis=1)
file_4 = pd.read_excel("data/achse/achse_des_guten_annotated_40000_items.xlsx")#.drop(["age"], axis=1)
# reddit
file_5 = annotate_to_dataframe('data/reddit/beziehungen.json')
file_6 = annotate_to_dataframe('data/reddit/BinIchDasArschloch.json')
file_7 = annotate_to_dataframe('data/reddit/DatingDE.json')
file_8 = annotate_to_dataframe('data/reddit/Eltern.json')
file_9 = annotate_to_dataframe('data/reddit/Ratschlag.json')
file_10 = annotate_to_dataframe('data/reddit/relationship_advice.json')
file_11 = annotate_to_dataframe('data/reddit/vaeter.json')

def group_ages(age: str) -> str:
    age = int(age)
    if age < 30:
        return 1
    else:
        return 2
    
def fix_sex(sex: str) -> str:
    if sex == "female":
        return "F"
    elif sex == "male":
        return "M"
    else:
        return sex
    
label_dict = {1: "Teens (<20)",
              2: "20s-30s",
              3: "40s-50s",
              4: "Early retirees (60-70)",
              5: "Seniors (70+)"}
    
def train_model():
    use_embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-german-cased")
    #model = AutoModelForMaskedLM.from_pretrained("bert-base-german-cased")

    use_vectors = np.array([use_embed([post]).numpy() for i, post in enumerate(raw_data["content"])])
    use_vectors = np.squeeze(use_vectors)

    with open("blogs_reddit_sentence_embeddings.npy", "wb") as f:
        np.save(f, use_vectors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
    parser.add_argument('-m', '--metric', type=str, required=True)
    args = parser.parse_args()
  
    raw_data = pd.concat([file_1, file_2, file_3, file_4, file_5, file_6, file_7, file_8, file_9, file_10, file_11])
    raw_data["age"] = raw_data["age"].apply(group_ages) 

    with open("blogs_reddit_sentence_embeddings.npy", "rb") as f:
        use_vectors = np.load(f)

    #use_mapper = umap.UMAP(metric=args.metric, random_state=42, n_neighbors=5).fit(use_vectors)
    #umap.plot.points(
    #    use_mapper,
    #    labels=np.array([row['age'] for x, row in raw_data.iterrows()]),
    #    background="black",
    #    width=1024,
    #    height=1024,
    #)

    #plt.show()

    #print(use_vectors)
    #model = tf.keras.Sequential([
    #    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    #    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    #    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
#])
    #model.compile(optimizer='adam',
    #          loss='sparse_categorical_crossentropy',
    #          metrics=['accuracy'])
    
    model = MLPClassifier()
    scaler = MinMaxScaler()
    X, y = use_vectors, raw_data["age"].apply(fix_sex)
    X, y = shuffle(X, y)
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #test_loss, test_acc = model.evaluate(X_test, y_test)
    #print('Test accuracy:', test_acc)
    print(classification_report(y_test, y_pred))
