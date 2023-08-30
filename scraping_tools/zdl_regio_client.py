import json
import matplotlib.pyplot as plt
import nltk
import operator
import pandas as pd
import random
import requests
import string
import nltk
from german_stopwords import stopwords_ as stopwords
from training_data_word_distribution import reddit_stopwords
from tqdm import tqdm
from pprint import pprint

def tokenize(sample: str) -> list:
    unique_tokens = list(set(nltk.word_tokenize(sample)))
    without_stop = [tok for tok in unique_tokens if tok.lower() not in stopwords]
    without_reddit = [tok for tok in without_stop if tok.lower() not in reddit_stopwords]
    return [tok for tok in without_reddit if tok not in string.punctuation and len(tok) <= 25]

def zdl_request(query: str, corpus: str = 'regional', by: str = 'areal', format: str = 'json'):
    zdl_base_url = "https://www.dwds.de/api/ppm"
    zdl_compiled_url = f"{zdl_base_url}?q={query}&corpus={corpus}&by={by}&format={format}"
    r = requests.get(zdl_compiled_url)
    return r.json()

def _get_most_frequent_areal(zdl_response: list[dict]) -> str:

    ppm_rel_val_dict = {}
    highest_val = 0
    return_areal = ""
    for areal in zdl_response:
        areal_ppm_rel = areal['ppm_rel']
        ppm_rel_val_dict[areal['areal']] = areal_ppm_rel
        if areal_ppm_rel > (highest_val):
            highest_val = areal_ppm_rel
            return_areal = areal['areal']
        else:
            continue
    sorted_ = sorted(ppm_rel_val_dict.items(), key=operator.itemgetter(1), reverse=True)
    return [item[0] for item in sorted_[:3]]

def get_most_frequent_areal_of_token(token: str) -> str:
    zdl_response = zdl_request(token)
    areal_top_3 = _get_most_frequent_areal(zdl_response)
    return areal_top_3

def text_sample_areal(text: str) -> str:

    match_counter = {
        'D-Nordwest': 0,
        'D-Nordost': 0,
        'D-Mittelwest': 0,
        'D-Mittelost': 0,
        'D-Südwest': 0,
        'D-Südost': 0,
        'unknown': 0
    }

    tokens = tokenize(text)

    for token in tqdm(tokens):
        try:
            a, b, c = get_most_frequent_areal_of_token(token)
        except (requests.exceptions.JSONDecodeError, ValueError):
            continue
        try:
            match_counter[a] += 1
            match_counter[b] += 0.5
            match_counter[c] += 0.25
            tqdm.write(f"{token}: {a} | {b} | {c}")
        except KeyError:
            match_counter['unknown'] += 1

    return match_counter


if __name__ == "__main__":
    
    with open("data/reddit/dating/beziehungen.json", "r", encoding="utf-8") as f:
        reddit = json.load(f)

    sample = str(reddit["data"][random.randint(0,1000)]["selftext"])
    print(sample)
    sample_frequency_dist = text_sample_areal(sample)
    pprint(sample_frequency_dist)
    plt.bar(sample_frequency_dist.keys(), sample_frequency_dist.values())
    plt.show()