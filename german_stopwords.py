import pandas as pd
from nltk.corpus import stopwords

stopwords_pd = pd.read_csv("https://zenodo.org/record/3995594/files/SW-DE-RS_v1-0-0_Datensatz.csv?download=1")
stopwords_ = []
for column in stopwords_pd.columns:
    stopwords_ += [_ for _ in stopwords_pd[column].tolist() if not isinstance(_, float)]
stopwords_ += stopwords.words('german')