from auxiliary import ABOUT
from corpus import DataCorpus, DataObject
from crawl_all_datasets import download_data
from data.gutenberg.gutenberg_to_dataobject import align_dicts
from models.zdl_vector_model import AREAL_DICT, ZDLVectorMatrix
from models.wiktionary_matrix import WiktionaryModel
from models.keras_cnn_implementation import *
from models.keras_regresssor_implementation import build_regressor
from scraping_tools.wiktionary_api import download_wiktionary

from tqdm import tqdm
from langdetect import detect, DetectorFactory, lang_detect_exception
from keras.backend import clear_session

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf

def __make_directories(path: str):
    os.makedirs(path, exist_ok=True)
    os.makedirs(f'{path}/achse', exist_ok=True)
    os.makedirs(f'{path}/annotation', exist_ok=True)
    os.makedirs(f'{path}/reddit', exist_ok=True)
    os.makedirs(f'{path}/reddit/locales', exist_ok=True)

def __init_parser() -> argparse.ArgumentParser:
    """
    creates an ArgumentParser which accepts a variety of arguments 
    that can be used to define what methods to execute.
    :returns: ArgumentParser()
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-z', '--zdl', action='store_true', help='build zdl vector database')
    parser.add_argument('-dd', '--download_data', action='store_true', help='download data resources')
    parser.add_argument('-dw', '--download_wikt', action='store_true', help='download wiktionary resources')
    parser.add_argument('-t', '--test', action='store_true', help='sets PATH to "test"')
    parser.add_argument('-b', '--build', action='store_true', help='builds a new corpus from scratch')
    parser.add_argument('-p', '--path', type=str, default='test')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-bw', '--build_wikt', action='store_true')
    parser.add_argument('-lw', '--load_wikt', action='store_true')
    parser.add_argument('-tr', '--train', action='store_true')
    parser.add_argument('-a', '--about', action='store_true')
    return parser

def __check_text_is_german(text: str) -> bool:
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

def __author_dict_to_dataobject() -> list[DataObject]:

    with open('data/gutenberg/data.json', 'r') as f:
        data = json.load(f)

    with open('data/gutenberg/author_dict.json', 'r') as f:
        author_dict = json.load(f)

    authors = align_dicts(data, author_dict)
    dataobjs = []

    for item in authors["books"]:
        if item["author_age"] != "":
            obj = DataObject(
                text=item["text"],
                author_age=int(item["author_age"]),
                author_gender=item["author_gender"],
                source="GUTENBERG"
                )
            dataobjs.append(obj)
    
    return dataobjs

def __reddit_locales_to_datacorpus(path: str = "data", corpus: DataCorpus = None):
    """ 
    finds all files in the reddit locale folder and adds them to a DataCorpus 
    :param path: defaults to data. can be changed to something else for testing, e.g. 'test'
    :param corpus: a DataCorpus object to which the data will be appended 
    :returns: the DataCorpus object with the added reddit data 
    """

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
                            if __check_text_is_german(text):
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
    reads reddit posts from annotated parquet file
    and adds to DataCorpus 
    :param path: path to parquest file with annotated posts
    :param corpus: DataCorpus object to append data to
    """
    reddit = pd.read_parquet(f'{path}/reddit/annotated_posts_2.parquet')

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

def __gutenberg_to_datacorpus(path: str, corpus: DataCorpus):
    gutenberg = __author_dict_to_dataobject()
    for item in tqdm(gutenberg):
        corpus.add_item(item)
    return corpus

def __build_corpus(data: DataCorpus, PATH: str) -> DataCorpus:
    print("loading reddit data from locales")
    data = __reddit_locales_to_datacorpus(PATH, data)
    print("loading achgut data")
    data = __achgut_to_datacorpus(PATH, data)
    print("loading reddit data from miscellaneous")
    data = __reddit_to_datacorpus(PATH, data)
    print("loading gutenberg data")
    data = __gutenberg_to_datacorpus(PATH, data)
    return data

def __to_num(L: list) -> list[float]:
    """ turns string labels into float """
    a = {
        "DE-MIDDLE-EAST": 0.0,
        "DE-MIDDLE-WEST": 1.0,
        "DE-NORTH-EAST": 2.0,
        "DE-NORTH-WEST": 3.0,
        "DE-SOUTH-EAST": 4.0,
        "DE-SOUTH-WEST": 5.0
        }
        
    b = {
        "finished_highschool": 0.0,
        "has_phd": 1.0,
        "has_apprentice": 2.0,
        "has_master": 3.0
        }

    c = {
        "female": 0.0,
        "male": 1.0,
        "f": 0.0,
        "m": 1.0
        }
        
    if L[0] in list(a.keys()):
        return [a[item] for item in L]
    elif L[0] in list(b.keys()):
        return [b[item] for item in L]
    elif L[0] in list(c.keys()): 
        return [c[item] for item in L]
    else:
        return L

def __build_zdl_vectors(data: DataCorpus):
    """
    This function uses ZDLVectorMatrix to generate ZDL vectors for texts in a DataCorpus.\n
    Each text will be tokenized, and each token gets embedded in a 6-dimensional vector.\n
    The token vectors are stored in an ordered array. \n
    So a ZDL vector representation of a text with 32 words will have this shape:\n
    (1, 32, 6).
    For model training, the vectors will be padded to the length of the longest document
    in the DataCorpus.\n
    The vectorization is performed in batches, and each batch's results will be stored in a 
    PARQUET file.\n
    :param data: DataCorpus object which contains labelled documents.
    """
    corpus_size = len(data)
    print(f"{corpus_size} items in DataCorpus.")
    dict_list: list[dict] = []  # will store dicts of format {ID: vector}
    batch_size = 1000  # amount of items to be vectorized in one batch
    for k in range(0, int(corpus_size/batch_size)+1):
        vector_database = pd.DataFrame()
        tqdm.write('Vectorizing Batch {}'.format(k+1))
        if k == 0:
            start = 0
        else:
            start = k*batch_size + 1
        
        end = (k+1) * batch_size

        if end > corpus_size:
             end = corpus_size - 1
        
        verbose = False
        if k > 30:
            verbose = True

        # if batch was already vectorized, skip
        check_file = os.path.isfile(f'test/ZDL/zdl_word_embeddings_batch_{k}.parquet')
        if check_file:
            print('Batch was already vectorized, skipping to next.')
            continue

        # call ZDLVectorMatrix class to perform vectorization
        sample_vectors = ZDLVectorMatrix(source=data[start:end], verbose=verbose).vectors
        dict_list.append(sample_vectors)

        for key in list(sample_vectors.keys()):
            sample_vectors[key] = sample_vectors[key].tolist()

        vector_database = pd.DataFrame(sample_vectors.items(), columns=['ID', 'embedding'])
        vector_database.to_parquet(f'test/ZDL/zdl_word_embeddings_batch_{k}.parquet')
    
    if dict_list:
        vectionary = dict_list[0]
        for sample in dict_list[1:]:
            vectionary.update(sample)
    else:
        print('all batches have been vectorized.')

def __zero_pad(X: list, maxVal: int) -> list:

    """
    Given the longest document in a corpus, this method will
    pad all data points to this size.
    :param X: raw training data, with vectors of varying shapes
    :param maxVal: the length of the longest document in the corpus
    """

    # iterate over each data point
    for i in range(len(X)):
        t = tf.convert_to_tensor(X[i], tf.float64)
        if len(t.shape) == 3:
            vector_length = t.shape[1]  # length of target vector
            diff = maxVal - vector_length   # amount of padding needed
            paddings = tf.constant([[0, 0], [0, diff], [0, 0]])
            t = tf.pad(t, paddings, "CONSTANT")
            
        else:
            t = tf.zeros([1, maxVal, 6], tf.float64)
        X[i] = t
    
    return X

def __maxval(X: list) -> int:

    maxVal = 0
    for x in X:
        x = np.array(x)
        if len(x.shape) == 1:
            continue
        length = x.shape[1]
        if length > maxVal:
            maxVal = length

    return maxVal

def __read_parquet(path: str) -> pd.DataFrame:
    # read parquet files back into dataframe
    directory_in_str = f"{path}/ZDL"
    directory = os.fsencode(directory_in_str)
    dataframe_list = []   
    for file in tqdm(os.listdir(directory)):
        filename = os.fsdecode(file)
        if filename.endswith(".parquet"): 
            temp = pd.read_parquet(f"{directory_in_str}/{filename}")
            dataframe_list.append(temp)
    vector_database = pd.concat(dataframe_list)
    return vector_database

if __name__ == "__main__":

    clear_session()  # clear any previous training sessions
    
    parser = __init_parser()
    args = parser.parse_args()

    # some assertions
    assert args.build_wikt ^ args.load_wikt ^ args.build ^ args.about, "you need to either load or build a wiktionary"
    assert args.test ^ (args.path != 'test')
    if args.about:
        remaining = [
            args.zdl,
            args.download_data,
            args.download_wikt,
            args.test,
            args.build,
            args.path != 'test',
            args.save,
            args.train
        ]
        assert not any(remaining), "-a (--about) can only be used on its own."
    
    if args.test:
        PATH = "test"
    else:
        PATH = args.path
    __make_directories(PATH)

    if args.about:
        print(ABOUT)

    if args.download_data:
        download_data(['achse', 'ortho', 'reddit_locales'], "test")

    if args.download_wikt:
        download_wiktionary()

    with open('data/wiktionary/wiktionary.json', 'r', encoding='utf-8') as f:
        wiktionary: dict = json.load(f)

    data = DataCorpus()

    # if we want to build corpus
    if args.build:
        data = __build_corpus(data, PATH)
        data.save_to_avro(f"{PATH}/corpus.avro")

    # read existing corpus
    data.read_avro(f'{PATH}/corpus.avro')
    print(len(data))

    if args.save:
        data.save_to_avro(f'{PATH}/corpus.avro')

    if args.build_wikt:
        wiktionary_matrix = WiktionaryModel(source=data)
        wiktionary_matrix.df_matrix.to_parquet('data/wiktionary/wiktionary.parquet')
    
    if args.load_wikt:
        wiktionary_matrix = WiktionaryModel('data/wiktionary/wiktionary.parquet')

    if args.zdl:
        __build_zdl_vectors(data=data)
        exit()

    vector_database = __read_parquet(PATH)

    if not args.train:
        exit()

    ids_: list[int] = []
    for item in data:
        if item.source in ("REDDIT, ACHGUT, GUTENBERG"):
            if item.author_gender not in ("N/A", "NONE", "", 0, "0", None):
                ids_.append(item.content['id'])
        else:
            continue
    
    # get corresponding zdl vectors for each id
    X_zdl = [vector_database[vector_database['ID'] == id].embedding.tolist() for id in ids_]
    # get corresponding wiktionary vectors
    X_wikt = [wiktionary_matrix[id] for id in ids_]

    # define target labels
    y = [data[id].author_gender for id in ids_]

    # shuffle the training data
    X, y = shuffle(ids_, y, random_state=3)
    
    # print label distribution
    for item in list(set(y)):
        print(f"{item}: {list(y).count(item)}")

    ids_train, ids_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42) #, stratify=y)
    
    # convert labels to tensor stack
    y_train = tf.stack(__to_num(y_train))
    y_test = tf.stack(__to_num(y_test))


    def RNN(n_inputs: int, n_outputs: int, X_train: tf.Tensor, X_test: tf.Tensor, y_train: list, y_test: list):
        model = rnn_model(n_inputs, n_outputs)
        print(model.summary())
        history = model.fit(X_train, y_train, 
                            epochs=64, 
                            verbose=True, 
                            validation_data=(X_test, y_test), 
                            batch_size=128,
                            use_multiprocessing=True,
                            workers=16)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def multiclass(n_inputs: int, n_outputs: int, X_train: tf.Tensor, X_test: tf.Tensor, y_train: list, y_test: list):
        model = multi_class_prediction_model(n_inputs, n_outputs)
        print(model.summary())

        history = model.fit(X_train, y_train, 
                            epochs=128, 
                            verbose=True, 
                            validation_data=(X_test, y_test), 
                            batch_size=32,
                            use_multiprocessing=True,
                            workers=16)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    def train_model(X: list, vectors: pd.DataFrame, ids_train: list, ids_test: list, y_train: list, y_test: list, source: str = "ZDL"):

        if source == "ZDL":
            Xtrain = [vectors[vectors['ID'] == id].embedding.tolist() for id in ids_train]
            Xtest = [vectors[vectors['ID'] == id].embedding.tolist() for id in ids_test]

            # get longest doc from corpus
            maxVal: int = __maxval(Xtest+Xtrain)

            # pad all vectors to that size
            X_train: list[tf.Tensor] = tf.stack(__zero_pad(Xtrain, maxVal))
            X_test: list[tf.Tensor] = tf.stack(__zero_pad(Xtest, maxVal))

            n_inputs, n_outputs = (1, maxVal, 6), y_train.shape[0]
            RNN(n_inputs, n_outputs, X_train, X_test, y_train, y_test)

        elif source == "Wikt":

            Xtrain = [vectors[id] for id in ids_train]
            Xtest = [vectors[id] for id in ids_test]
            X_train: list[tf.Tensor] = tf.stack(Xtrain)
            X_test: list[tf.Tensor] = tf.stack(Xtest)

            n_inputs, n_outputs = (27, ), y_train.shape[0]
            multiclass(n_inputs, n_outputs, X_train, X_test, y_train, y_test)

    train_model(
        X=X_wikt, 
        vectors=wiktionary_matrix, 
        ids_train=ids_train, 
        ids_test=ids_test, 
        y_train=y_train, 
        y_test=y_test,
        source="Wikt")

    ### BINARY PREDICTION (e.g. gender)
    def binary():
        model = binary_prediction_model(n_inputs)
        print(model.summary())

        history = model.fit(X_train, y_train, 
                            epochs=256, 
                            verbose=True, 
                            validation_data=(X_test, y_test), 
                            batch_size=32,
                            use_multiprocessing=True,
                            workers=6)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))

    # binary()


    #multiclass()    
    exit() 