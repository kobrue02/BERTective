import json
import matplotlib.pyplot as plt
import nltk
import string

from german_stopwords import stopwords_ as stopwords

with open("data/reddit/dating/beziehungen.json", "r", encoding="utf-8") as f:
    reddit = json.load(f)

all_text = ""

for item in reddit["data"]:
    all_text += str(item["selftext"])

tokens = nltk.word_tokenize(all_text)

tokens = [tok for tok in tokens if tok not in stopwords]
tokens = [tok.lower() for tok in tokens if tok not in string.punctuation]

word_frequency_raw = {}
for token in tokens:
    if token not in word_frequency_raw:
        word_frequency_raw[token] = 1
    else:
        word_frequency_raw[token] += 1

word_frequency = {}
for word in word_frequency_raw.keys():
    if word_frequency_raw[word] > 150:
        word_frequency[word] = word_frequency_raw[word]

reddit_stopwords = word_frequency.keys()

if __name__ == "__main__":
    print(word_frequency)
    plt.rcParams.update({'font.size': 7})
    plt.bar(word_frequency.keys(), word_frequency.values())
    plt.xticks(rotation=90)
    plt.show()