import pandas as pd
from urllib.error import URLError
from nltk.corpus import stopwords

try:
    stopwords_pd = pd.read_csv("https://zenodo.org/record/3995594/files/SW-DE-RS_v1-0-0_Datensatz.csv?download=1")
except URLError:
    raise URLError('Your python installation is missing SSL certificates, \
                   probably because you installed it using Brew. \
                   Solution here: https://stackoverflow.com/questions/44649449/brew-installation-of-python-3-6-1-ssl-certificate-verify-failed-certificate/44649450#44649450')

stopwords_ = []
for column in stopwords_pd.columns:
    stopwords_ += [_ for _ in stopwords_pd[column].tolist() if not isinstance(_, float)]
stopwords_ += stopwords.words('german')