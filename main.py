import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from src.auxiliary import ABOUT, LOGO, MALE_CHARS, FEMALE_CHARS
from src.exceptions import *
from src.corpus import DataCorpus, DataObject
from src.crawl_all_datasets import download_data
from data.gutenberg.gutenberg_to_dataobject import align_dicts
from models.zdl_vector_model import AREAL_DICT, ZDLVectorMatrix, ZDLVectorModel
from models.wiktionary_matrix import WiktionaryModel
from models.keras_cnn_implementation import *
from models.keras_regresssor_implementation import build_regressor
from models.error_and_ortho_matrix import OrthoMatrixModel
from models.bare_statistic_ratios import Statistext
from scraping_tools.wiktionary_api import download_wiktionary

from tqdm import tqdm
from typing import Union
from langdetect import detect, DetectorFactory, lang_detect_exception
from keras.backend import clear_session
from keras.callbacks import EarlyStopping
from keras.models import load_model
from typing import Any

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import seaborn as sns

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def __make_directories(path: str) -> None:
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
    parser.add_argument('-bz', '--build_zdl', action='store_true', help='build zdl vector database')
    parser.add_argument('-dd', '--download_data', action='store_true', help='download data resources')
    parser.add_argument('-dw', '--download_wikt', action='store_true', help='download wiktionary resources')
    parser.add_argument('-t', '--test', action='store_true', help='sets PATH to "test"')
    parser.add_argument('-b', '--build', action='store_true', help='builds a new corpus from scratch')
    parser.add_argument('-p', '--path', type=str, default='test')
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('-bw', '--build_wikt', action='store_true')
    parser.add_argument('-bs', '--build_stats', action='store_true')
    parser.add_argument('-tr', '--train', action='store_true')
    parser.add_argument('-a', '--about', action='store_true')
    parser.add_argument('-q', '--query', type=str, default=None)
    parser.add_argument('-bo', '--build_ortho', action='store_true')
    parser.add_argument('-age', '--age', action='store_true')
    parser.add_argument('-gender', '--gender', action='store_true')
    parser.add_argument('-regio', '--regiolect', action='store_true')
    parser.add_argument('-edu', '--education', action='store_true')
    parser.add_argument('-f', '--feature', type=str, default="ortho")
    parser.add_argument('-m', '--model', type=str, default='multiclass')
    parser.add_argument('-src', '--source', type=str, nargs='*', default=["ACHGUT", "REDDIT", "GUTENBERG"])
    parser.add_argument('-pr', '--predict', type=str, default=None)
    parser.add_argument('-n', '--number', type=int, default=4000)
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

def __reddit_locales_to_datacorpus(path: str = "data", corpus: DataCorpus = None) -> DataCorpus:
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
    
def __achgut_to_datacorpus(path: str, corpus: DataCorpus) -> DataCorpus:
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

def __reddit_to_datacorpus(path: str, corpus: DataCorpus) -> DataCorpus:
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

def __gutenberg_to_datacorpus(path: str, corpus: DataCorpus) -> DataCorpus:
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

def __to_num(L: list, key: str) -> list[float]:
    """ turns string labels into float """
    labels = {
                'author_regiolect': {
                        "DE-MIDDLE-EAST": 0.0,
                        "DE-MIDDLE-WEST": 1.0,
                        "DE-NORTH-EAST": 2.0,
                        "DE-NORTH-WEST": 3.0,
                        "DE-SOUTH-EAST": 4.0,
                        "DE-SOUTH-WEST": 5.0
                        },
                'author_education': {
                        "finished_highschool": 0.0,
                        "has_phd": 1.0,
                        "has_apprentice": 2.0,
                        "has_master": 3.0
                        },
                'author_gender': {
                    "female": 0.0,
                    "male": 1.0,
                    "f": 0.0,
                    "m": 1.0
                    }
    }
    if key == "author_age":
        def __age(age: str) -> float:
            return float(int(int(age)/10)) - 1.0
        return [__age(str(i)) for i in L]
    else:
        return [labels.get(key)[i] for i in L]

def __num_to_str(L: list[float], key: str) -> list[str]:
    """ convert the numerical labels back to their true names """
    
    labels = {
                'author_regiolect': {
                    0.0: 'DE-MIDDLE-EAST',
                    1.0: 'DE-MIDDLE-WEST',
                    2.0: 'DE-NORTH-EAST',
                    3.0: 'DE-NORTH-WEST',
                    4.0: 'DE-SOUTH-EAST',
                    5.0: 'DE-SOUTH-WEST'
                },
                'author_education': {
                    0.0: 'finished_highschool',
                    1.0: 'has_phd',
                    2.0: 'has_apprentice',
                    3.0: 'has_master'
                },
                'author_gender': {
                    0.0: 'female',
                    1.0: 'male'
                },
                'author_age': {
                    0.0: '10-20',
                    1.0: '20-30',
                    2.0: '30-40',
                    3.0: '40-50',
                    4.0: '50-60',
                    5.0: '60-70',
                    6.0: '70-80',
                    7.0: '80-90'
                }
            }
    if isinstance(L[0], str):
        for i in range(len(L)):
            if L[i] == "f":
                L[i] = "female"
            elif L[i] == "m":
                L[i] = "male"
            else:
                pass
        return L
    
    elif isinstance(L[0], int) and key == "author_age":
        def __age(age: int) -> str:
            a = str(age)[0]
            b = int(a)*10
            return f"{b}-{b+10}"
        return [__age(i) for i in L]
    else:
        return [labels.get(key)[i] for i in L]

def __build_zdl_vectors(data: DataCorpus) -> None:
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

def __get_ortho_embedding(text: str) -> dict[str, list]:

    ortho = OrthoMatrixModel()

    ancient = ortho.find_ortho_match_in_text(text, 'ancient').tolist()
    revolutionized = ortho.find_ortho_match_in_text(text, 'revolutionized').tolist()
    modern = ortho.find_ortho_match_in_text(text, 'modern').tolist()
    error = ortho.find_error_match_in_text(text, 'error').tolist()
    correct = ortho.find_error_match_in_text(text, 'correct').tolist()

    return {
            'embedding_ancient': ancient,
            'embedding_revolutionized': revolutionized,
            'embedding_modern': modern,
            'embedding_error': error,
            'embedding_correct': correct
        }

def __get_wiktionary_embedding(text: str) -> np.ndarray:
    wm = WiktionaryModel()
    _, vector = wm.get_matches(text)
    return np.array(vector)

def __get_zdl_embedding(text: str) -> np.ndarray:
    with open('vectors/zdl_vector_dict.json', 'r', encoding='utf-8') as f:
        vectionary: dict = json.load(f)
    vector, _ = ZDLVectorModel._vectorize_sample(text, vectionary, verbose=False)
    return vector

def __get_statistical_embedding(text: str) -> np.ndarray:
    stats = Statistext(text)
    return stats.all_stats
    
def __build_ortho_matrix(data: DataCorpus) -> dict[str, dict[str, np.ndarray]]:
    """
    takes a DataCorpus as input and calculates an orthography/vector embedding for every text.
    5 embeddings are generated for each text sample:
    ancient: matches with ancient orthógraphy set
    revolutionized: matches with revolutionized orthography set
    modern: matches with current orthography set
    error: common spelling errors
    correct: words that were spelt correct but are commonly misspelt
    these 5 embeddings are stored as a dict in a dict that has DataObject ID as key and embedding dict as value
    """
    ortho = OrthoMatrixModel()
    corpus_size = len(data)
    matrix = {}
    for n in tqdm(range(corpus_size)):

        ID = data[n].content['id']
        text = data[n].text

        matrix[ID] = __get_ortho_embedding(text)

    return matrix

def __build_statistical_matrix(data: DataCorpus) -> dict[str, dict[str, float]]:
    """
    takes a DataCorpus as input and calculates bare statistic features for each text.
    """
    corpus_size = len(data)
    matrix = {}
    for n in tqdm(range(corpus_size)):

        ID = data[n].content['id']
        text = data[n].text

        statistecs = Statistext(text)
        matrix[ID] = statistecs.all_stats
    return matrix

def __zero_pad(X: list, maxVal: int) -> list[tf.Tensor]:

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
        length = x.shape[0]
        if length > maxVal:
            maxVal = length

    return maxVal

def __read_parquet(path: str) -> pd.DataFrame:
    # read parquet files back into dataframe
    print("Loading ZDL vectors from disk...")
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

def __get_query(data: DataCorpus, query: str) -> list[DataObject]:

    label = query.split("=")[0]
    value = query.split("=")[1]
    Q = {label: value}
    items = data.query(Q)

    return items

def __plot_items(items: list[DataObject]) -> None:

    age_dist = {}
    gender_dist = {}
    regiolect_dist = {}
    education_dist = {}

    for item in items:

        # age
        if item.author_age in age_dist:
            age_dist[item.author_age] += 1
        else:
            age_dist[item.author_age] = 1
        
        # gender
        if item.author_gender in gender_dist:
            gender_dist[item.author_gender] += 1
        else:
            gender_dist[item.author_gender] = 1
        
        # regiolect
        if item.author_regiolect in regiolect_dist:
            regiolect_dist[item.author_regiolect] += 1
        else:
            regiolect_dist[item.author_regiolect] = 1

        # education
        if item.author_education in education_dist:
            education_dist[item.author_education] += 1
        else:
            education_dist[item.author_education] = 1

    sns.barplot(x=list(regiolect_dist.keys()), y=list(regiolect_dist.values()))
    plt.show()
        
def __evaluate(model: Sequential, X_test: list[float], y_test: list[str], key: str) -> str:
    y_pred = model.predict(X_test)
    # set the labels and predictions to same type
    # so that we can generate a classification report
    y_pred = np.round(y_pred)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = __num_to_str([y for y in y_test], key)
    y_pred = __num_to_str([y for y in y_pred], key)
    report = classification_report(y_test, y_pred)
    return report

def __setup() -> argparse.Namespace:
    
    parser = __init_parser()
    args = parser.parse_args()

    # some assertions
    assert args.test ^ (args.path != 'test')
    assert args.model in ["rnn", "multiclass", "binary"]

    if args.about:
        remaining = [
            args.build_zdl,
            args.download_data,
            args.download_wikt,
            args.test,
            args.build,
            args.path != 'test',
            args.save,
            args.train
        ]
        assert not any(remaining), "-a (--about) can only be used on its own."
    return args

def __concat_vectors(
        zdl_vector: np.ndarray, 
        ortho_vector: np.ndarray, 
        wiktionary_vector: np.ndarray, 
        statistical_array: np.ndarray,
        maxVal: int = 0) -> np.ndarray:
    
    """
    Concatenate 4 arrays of different shapes and dimensions to generate 
    one 2D array representing all features.
    :param zdl_vector: numpy array containing a ZDL vector, usually of shape (1, 1931, 6)
    :ortho_vector: numpy array representing orthography matrix. (5, 96)
    :wiktionary_vector: numpy array of shape (27, ) which represents wiktionary vocabulary use
    :statistical_array: numpy array that contains bare statistical features, shape TBD
    """

    # Find the maximum dimensions for padding
    if maxVal == 0:
        max_rows = max(zdl_vector.shape[0], ortho_vector.shape[0], wiktionary_vector.shape[0], statistical_array.shape[0])
    else:
        max_rows = max(maxVal, ortho_vector.shape[0], wiktionary_vector.shape[0], statistical_array.shape[0])
    
    try:
        max_cols = max(zdl_vector.shape[1], ortho_vector.shape[1])
    except IndexError:
        print(zdl_vector)
        exit()

    # Create new arrays with the maximum dimensions and fill with zeros
    combined_array = np.zeros((max_rows, max_cols))

    # Copy the values from the original arrays to the new arrays
    combined_array[:zdl_vector.shape[0], :zdl_vector.shape[1]] = zdl_vector
    combined_array[:ortho_vector.shape[0], :ortho_vector.shape[1]] = ortho_vector
    combined_array[:wiktionary_vector.shape[0], :1] = wiktionary_vector.reshape(-1, 1)  # Reshape and copy
    combined_array[:statistical_array.shape[0], :1] = statistical_array.reshape(-1, 1)  # Reshape and copy
    return combined_array

def __preprocess(sample: str, maxVal: int = 0) -> tuple[list[tf.Tensor], list[tf.Tensor]]:
    """
    Preproces a text and extract all features using the respective methods.
    Returns one list of tensors with all mapped features, and one tensor list with only ZDL vectors.
    """
    print('Analysing the text...')
    zdl_vector = __get_zdl_embedding(sample)
    ortho_vector = np.array(list(__get_ortho_embedding(sample).values()))
    wiktionary_vector = __get_wiktionary_embedding(sample)
    statistical_array = __get_statistical_embedding(sample)

    combined_array = __concat_vectors(zdl_vector, ortho_vector, wiktionary_vector, statistical_array, maxVal)
    
    print(f"The text input has been embedded into a vector of the shape {combined_array.shape}.")
    for_inference = tf.stack([combined_array])
    return for_inference, __zero_pad([zdl_vector], 1931)

def __concat_all_corpus_features(data: DataCorpus, ids: list[int]) -> dict[int, np.ndarray]:
    zdl_vector_database = __read_parquet(PATH)
    maxVal: int = __maxval([np.array([x for x in embedding]) for embedding in zdl_vector_database.embedding.tolist()])
    with open(f'vectors/statistical_matrix.json', 'r') as f:
            statistics: dict[str, dict] = json.load(f)
    with open('vectors/orthography_matrix.json', 'r') as f:
            orthoMatrix: dict[str, dict] = json.load(f)
    wiktionary_matrix = WiktionaryModel('data/wiktionary/wiktionary.parquet')
    features = {}
    print("Mapping all features into one vector...")
    for n in tqdm(ids):

        ID = data[n].content['id']

        try:
            zdl_vector = np.array([x for x in zdl_vector_database.loc[zdl_vector_database['ID'] == ID, 'embedding'].iloc[0]])
            if zdl_vector.size == 0:
                zdl_vector = np.zeros((maxVal, 6))
        except IndexError:
            zdl_vector = np.zeros((maxVal, 6))
        ortho_vector = np.array(list(orthoMatrix[str(ID)].values()))
        wikt_vector = wiktionary_matrix[ID]
        stat_vector = np.array(list(statistics[str(ID)].values()))
        combined_vector = __concat_vectors(zdl_vector, ortho_vector, wikt_vector, stat_vector, maxVal)
        features[ID] = combined_vector
    #print("Storing complete features in npy files...")
    #os.makedirs("vectors/mapped_features", exist_ok=True)
    #try:
    #    h5f = h5py.File(f'vectors/mapped_features/data.h5', 'w')
    #    for id_, embedding in tqdm(features.items()):
    #        h5f.create_dataset(str(id_), data=embedding)
    #    h5f.close()
    #except: pass
    return features, maxVal

def __get_training_data(feature: str, ids_: list[int]) -> tuple[list[np.ndarray], str, Any]:
    maxVal = 1931
    if feature.capitalize() == "Ortho":
        ortho_path = 'vectors/orthography_matrix.json'
        if not os.path.exists(ortho_path):
            raise MissingTrainingData("It seems that the orthography matrix has not been generated yet."
                                      "Please do so, using -bo")
        with open(ortho_path, 'r') as f:
            orthoMatrix: dict[str, dict] = json.load(f)
        X = [np.array(list(orthoMatrix[str(ID)].values())) for ID in ids_]
        source = "Ortho"
        vectors = orthoMatrix

    elif feature.capitalize() == "Stat":
        stat_path = 'vectors/statistical_matrix.json'
        if not os.path.exists(stat_path):
            raise MissingTrainingData("It seems that the statistical features have not been generated yet."
                                      "Please do so, using -bs")
        with open(stat_path, 'r') as f:
            statistext: dict[str, dict] = json.load(f)
        X = [np.array(list(statistext[str(ID)].values())) for ID in ids_]
        source = "Stat"
        vectors = statistext

    elif feature.lower() == "zdl":
        if not os.path.exists(f"{PATH}/ZDL"):
            raise MissingTrainingData("It seems that the ZDL vectors have not been generated yet."
                                      "Please do so, using -bz")
        vector_database = __read_parquet(PATH)
        X = [vector_database[vector_database['ID'] == id].embedding.tolist() for id in ids_]
        source = "ZDL"
        vectors = vector_database

    elif feature.lower() == "wikt":
        wikt_path = 'data/wiktionary/wiktionary.parquet'
        if not os.path.exists('data/wiktionary/wiktionary.parquet'):
            raise MissingTrainingData("It seems that the Wiktionary Matrix has not been generated yet."
                                      "Please do so, using -bw")
        wiktionary_matrix = WiktionaryModel(wikt_path)
        X = [wiktionary_matrix[id] for id in ids_]
        source = "Wikt"
        vectors = wiktionary_matrix

    elif feature.lower() == "all":
        X, maxVal = __concat_all_corpus_features(data, ids_)
        source = "all"
        vectors = None
    return X, source, vectors, maxVal

def __get_text_from_arg(input_str: str) -> str:
    pattern = r"""^(?:[a-z]:)?[\/\\]{0,2}(?:[.\/\\ ](?![.\/\\\n])|[^<>:\"|?*.\/\\ \n])+$"""
    is_path = re.search(pattern, input_str)
    if is_path:
        if not os.path.exists(input_str):
             raise ParsedPathNotExistError("Your input '{}' looks like a path, but the file doesn't exist.".format(input_str))
        print("reading file...")
        with open(input_str, 'r', encoding='utf-8') as f:
            input_str = f.read()
    return input_str

def __predict(input_arg: str, model_features: str = 'all') -> dict[str, dict[str, float]]:
    """
    Inference using all pretrained models.
    Loads models from disk and uses them to predict all author profile features.
    Returns dict with results.
    """
    
    text = __get_text_from_arg(input_arg)
    data, zdl_vector = __preprocess(text, 1931)
    features = ['author_gender', 'author_age', 'author_education', 'author_regiolect']
    profile = {}
    for feature in features:
        print(f'Infering {feature}...')
        try:
            if feature == 'author_regiolect':
                reconstructed_model = load_model(f'models/trained_models/ZDL_features_{feature}.model')
                pred = reconstructed_model.predict(tf.stack(zdl_vector))
            else:
                reconstructed_model = load_model(f'models/trained_models/fully_mapped_features_{feature}.model')
                pred = reconstructed_model.predict(data)
        except (FileNotFoundError, OSError):
            print(f'Did not find a model that predicts {feature}.')
            continue
        
        y_pred = np.round(pred)
        confidence = max(pred[0])
        output_label = np.argmax(y_pred, axis=1)
        output_label = __num_to_str(output_label, feature)

        profile[feature] = {'label': output_label[0], 'confidence': confidence}
    return profile

def __print_profile() -> None:
    """
    Loads pretrained models to make prediction for input text.
    This prediction is printed to the console and nothing is returned.
    """
    pred = __predict(args.predict)

    gender_predicted = 'author_gender' in list(pred.keys())
    edu_predicted = 'author_education' in list(pred.keys())
    regio_predicted = 'author_regiolect' in list(pred.keys())
    age_predicted = 'author_age' in list(pred.keys())

    if gender_predicted:
        gender = pred['author_gender']['label']
        gender_conf = pred['author_gender']['confidence']
        adj = '' if gender_conf > 0.9 else 'probably '
        if gender == "male":
            print(random.choice(MALE_CHARS))
        else:
            print(random.choice(FEMALE_CHARS))
        print(f"""The author is {adj}{gender}. Confidence: {gender_conf:.1%}.""")
        pronoun_possessive = "His" if gender=='male' else "Her"
        pronoun_1p = "He" if gender=="male" else "She"

    if edu_predicted:
        edu = pred['author_education']['label']
        edu_conf = pred['author_education']['confidence']
        adj = '' if edu_conf > 0.9 else 'most likely '
        print(f"{pronoun_possessive} degree of education is {adj}{edu}. Confidence: {edu_conf:.1%}.")

    if regio_predicted:
        regio = pred['author_regiolect']['label']
        regio_conf = pred['author_regiolect']['confidence']
        adj = '' if regio_conf > 0.9 else 'probably '
        print(f"{pronoun_1p} {adj}comes from {regio}. Confidence: {regio_conf:.1%}.")

    if age_predicted:
        age = pred['author_age']['label']
        age_conf = pred['author_age']['confidence']
        print(f"{pronoun_1p} is approximately {age} years old. Confidence: {age_conf:.1%}.")

    exit()

def __get_applicable_ids(data: DataCorpus, feature: dict, F: argparse.Namespace):
    """
    
    """
    ids_: list[int] = []
    for item in data:
        if item.source in args.source:
            if item.content[feature[F]] not in ("N/A", "NONE", "", 0, "0", None):
                ids_.append(item.content['id'])
        else:
            continue

    if args.number < len(ids_):
        ids_ = random.sample(ids_, args.number)
    return ids_

def __prepare_training(data: DataCorpus, args: argparse.Namespace) -> tuple[list, str, list, int, list, list, list, list, argparse.Namespace, str]:
    """
    Takes a DataCorpus and argparse Namespace as input,
    and prepares a training session by generating the required data.
    """
    
    feature = {args.education: 'author_education',
               args.age: 'author_age',
               args.regiolect: 'author_regiolect',
               args.gender: 'author_gender'}
    F = args.education or args.age or args.regiolect or args.gender
    ids_ = __get_applicable_ids(data, feature, F)
    
     # define target labels
    y = [data[id].content[feature[F]] for id in ids_]

    # get training data
    X_data, source, vectors, MAXVAL = __get_training_data(args.feature, ids_)
    
    # shuffle the training data
    X, y = shuffle(ids_, y, random_state=3)
    
    # print label distribution
    print("Label distribution:")
    for item in list(set(y)):
        print(f"{item}: {list(y).count(item)}")
    if feature[F] == "author_gender":
        print("Note that f and female as well as and m and male will be merged into one label.")

    ids_train, ids_test, y_train, y_test_ = train_test_split(
                X, y, test_size=0.2, random_state=42) #, stratify=y)
    
    # convert labels to tensor stack
    y_train = tf.stack(__to_num(y_train, feature[F]))
    y_test = __to_num(y_test_, feature[F])

    return X_data, source, vectors, MAXVAL, ids_train, ids_test, y_train, y_test, feature, F

def RNN(n_inputs: int, n_outputs: int, X_train: tf.Tensor, X_test: tf.Tensor, y_train: list, y_test: list):
    model = rnn_model(n_inputs, n_outputs)
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='auto')
    print(model.summary())
    try:
        history = model.fit(X_train, y_train, 
                            epochs=128, 
                            verbose=True, 
                            validation_data=(X_test, y_test), 
                            batch_size=128,
                            use_multiprocessing=True,
                            workers=16,
                            callbacks = [early_stopping])
    except KeyboardInterrupt:
        pass

    return model

def multiclass(n_inputs: int, n_outputs: int, X_train: tf.Tensor, X_test: tf.Tensor, y_train: list, y_test: list):
    model = multi_class_prediction_model(n_inputs, n_outputs)
    print(model.summary())
    early_stopping = EarlyStopping(monitor='val_loss', patience=8, mode='auto')
    try:
        history = model.fit(X_train, y_train, 
                            epochs=128, 
                            verbose=True, 
                            validation_data=(X_test, y_test), 
                            batch_size=64,
                            use_multiprocessing=True,
                            workers=16,
                            callbacks = [early_stopping])
    except KeyboardInterrupt:
        pass

    return model

def binary(n_inputs: int, n_outputs: int, X_train: tf.Tensor, X_test: tf.Tensor, y_train: list, y_test: list):
    model = binary_prediction_model(n_inputs)
    print(model.summary())

    history = model.fit(X_train, y_train, 
                        epochs=256, 
                        verbose=True, 
                        validation_data=(X_test, y_test), 
                        batch_size=32,
                        use_multiprocessing=True,
                        workers=6)

    return model

def train_model(X: Union[list,dict], 
                    vectors, 
                    ids_train: list, 
                    ids_test: list, 
                    y_train: list, 
                    y_test_: list, 
                    source: str = "Ortho",
                    model_type: str = "multiclass",
                    max_val: int = None) -> tuple[Sequential, list[float], list[str]]:
        
        model_select = {
            "rnn": RNN,
            "multiclass": multiclass,
            "binary": binary
        }
        model_ = model_select.get(model_type)
        y_test = tf.stack(y_test_)
        if source == "ZDL":
            Xtrain = [vectors[vectors['ID'] == id].embedding.tolist() for id in ids_train]
            Xtest = [vectors[vectors['ID'] == id].embedding.tolist() for id in ids_test]

            # pad all vectors to same size
            X_train: list[tf.Tensor] = tf.stack(__zero_pad(Xtrain, max_val))
            X_test: list[tf.Tensor] = tf.stack(__zero_pad(Xtest, max_val))

            n_inputs, n_outputs = (1, max_val, 6), y_train.shape[0]
            model = model_(n_inputs, n_outputs, X_train, X_test, y_train, y_test)
            os.makedirs('models/trained_models', exist_ok=True)
            model.save(f'models/trained_models/ZDL_features_{feature[F]}.model')

        elif source == "Wikt":

            Xtrain = [vectors[id] for id in ids_train]
            Xtest = [vectors[id] for id in ids_test]
            X_train: list[tf.Tensor] = tf.stack(Xtrain)
            X_test: list[tf.Tensor] = tf.stack(Xtest)

            n_inputs, n_outputs = (27, ), y_train.shape[0]
            model = model_(n_inputs, n_outputs, X_train, X_test, y_train, y_test)
        
        elif source == "Ortho":

            Xtrain = [list(vectors[str(ID)].values()) for ID in ids_train]
            Xtest = [list(vectors[str(ID)].values()) for ID in ids_test]
            X_train: list[tf.Tensor] = tf.stack(Xtrain)
            X_test: list[tf.Tensor] = tf.stack(Xtest)

            n_inputs, n_outputs = (5, 96), y_train.shape[0]
            model = model_(n_inputs, n_outputs, X_train, X_test, y_train, y_test)

        elif source == "Stat":

            Xtrain = [list(vectors[str(ID)].values()) for ID in ids_train]
            Xtest = [list(vectors[str(ID)].values()) for ID in ids_test]
            X_train: list[tf.Tensor] = tf.stack(Xtrain)
            X_test: list[tf.Tensor] = tf.stack(Xtest)

            n_inputs, n_outputs = (4, ), y_train.shape[0]
            model = multiclass(n_inputs, n_outputs, X_train, X_test, y_train, y_test)

        elif source == "all":
            Xtrain = [X.get(ID) for ID in ids_train]
            Xtest = [X.get(ID) for ID in ids_test]
            X_train: list[tf.Tensor] = tf.stack(Xtrain)
            X_test: list[tf.Tensor] = tf.stack(Xtest)
            n_inputs, n_outputs = (max_val, 96), y_train.shape[0]
            model = model_(n_inputs, n_outputs, X_train, X_test, y_train, y_test)
            os.makedirs('models/trained_models', exist_ok=True)
            model.save(f'models/trained_models/fully_mapped_features_{feature[F]}.model')

        return model, X_test, y_test_

if __name__ == "__main__":

    clear_session()  # clear any previous training sessions
    args = __setup()
    print(LOGO)
    if args.test:
        PATH = "test"
    else:
        PATH = args.path
    __make_directories(PATH)

    if args.about:
        # help
        print(ABOUT)

    if args.download_data:
        # download all data resources
        download_data(['achse', 'ortho', 'reddit_locales'], "test")

    if args.download_wikt:
        # download wiktionary with scraper
        download_wiktionary()

    data = DataCorpus()

    if args.predict != None:
        __print_profile()

    # if we want to build corpus
    if args.build:
        data = __build_corpus(data, PATH)
        data.save_to_avro(f"{PATH}/corpus.avro")

    # read existing corpus
    data.read_avro(f'{PATH}/corpus.avro')
    print(f"The dataset contains {len(data)} items.")

    if args.query != None:
        items = __get_query(data, args.query)
        __plot_items(items)

    if args.save:
        data.save_to_avro(f'{PATH}/corpus.avro')

    if args.build_wikt:
        wiktionary_matrix = WiktionaryModel(source=data)
        wiktionary_matrix.df_matrix.to_parquet('data/wiktionary/wiktionary.parquet')
    
    if args.build_zdl:
        __build_zdl_vectors(data=data)

    if args.build_ortho:
        ortho_matrix = __build_ortho_matrix(data)
        with open(f'vectors/orthography_matrix.json', 'w') as f:
            json.dump(ortho_matrix, f)
        exit()

    if args.build_stats:
        statistext = __build_statistical_matrix(data)
        with open(f'vectors/statistical_matrix.json', 'w') as f:
            json.dump(statistext, f)
        exit()

    if args.train:
        X_data, source, vectors, MAXVAL, \
            ids_train, ids_test, y_train, y_test, feature, F = __prepare_training(data, args)
        model, X_test, y_test = train_model(
                                    X=X_data, 
                                    vectors=vectors, 
                                    ids_train=ids_train, 
                                    ids_test=ids_test, 
                                    y_train=y_train, 
                                    y_test_=y_test,
                                    source=source,
                                    model_type=args.model,
                                    max_val=MAXVAL
                                )
        report = __evaluate(model, X_test, y_test, key=feature[F])
        print(report)

    exit() 