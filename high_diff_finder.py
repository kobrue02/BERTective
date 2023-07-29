import requests
import json
import operator
import spacy
from time import sleep
from tqdm import tqdm

from zdl_regio_client import zdl_request

import spacy

nlp = spacy.load('de_core_news_md')

word_diff = {}

with open("data/ZDL/wordlist-german.txt", "r", encoding="utf-8") as f:
    wordlist_german_raw = [line.replace("\n", "") for line in f.readlines()]

def lemma():
    wordlist_german = []
    with open("data/ZDL/wordlist-lemmas_2.txt", "w", encoding="UTF-8") as f:
        for i, word in enumerate(tqdm(wordlist_german_raw)):
            last_word = wordlist_german_raw[i-1]
            cutoff = int(0.75 * len(last_word))
            if word[:cutoff] == last_word[:cutoff]:
                continue
            else:
                wordlist_german.append(word)
                f.write(f"{word}\n")

with open("data/ZDL/wordlist-lemmas_2.txt", "r", encoding="utf-8") as f:
    wordlist_german = [line.replace("\n", "") for line in f.readlines()]

print(len(wordlist_german))

for word in tqdm(wordlist_german):
    try:
        r = zdl_request(word)
    except:
        sleep(10)
        continue
    ppm_dict = {item["areal"]: item["ppm_rel"] for item in r}
    sorted_ = sorted(ppm_dict.items(), key=operator.itemgetter(1), reverse=True)
    if len(sorted_) < 2:
        continue
    max_val = sorted_[0][1]
    second_val = sorted_[1][1]
    diff = max_val - second_val
    word_diff[word] = diff
    tqdm.write(f"{word}: {diff}")


with open("data/ZDL/diff.json", "r", encoding="utf-8") as f:
    json.dump({"diffs": word_diff})