import requests
import time
import json

from bs4 import BeautifulSoup
from pprint import pprint

WIKI_LIST = [
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Abkürzungen_im_Internet',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Anglizismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Arabismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Aufwertung',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Binomiale',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Biologie/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Chemie/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Chinesische_Wörter_in_der_deutschen_Sprache',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/DDR-Sprache',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Gallizismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Geowissenschaften',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Geschichte',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Hispanismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Japanische_Wörter_in_der_deutschen_Sprache',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Jägersprache',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Mathematik/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Medizin/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Militär/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Musik/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Netzjargon',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Ornithologie/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Pharmazie',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Philosophie',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Physik/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Politik',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Rechtswissenschaften',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Religion',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Romanismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Slawismen',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Weinbau',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Wirtschaft',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Griechische_Präfixe',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Griechische_Suffixe',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Informationstechnik',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Lateinische_Präfixe',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Lateinische_Suffixe',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Linguistik/Fachwortliste',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Philatelie',
        'https://de.download_wiktionary.org/wiki/Verzeichnis:Deutsch/Sprichwörter'
    ]


def call_url(url: str):
    r = requests.get(url)
    return r.text

def find_word_list(url: str) -> list[str]:
    html = call_url(url)
    soup = BeautifulSoup(html, features="lxml")
    parser_output = soup.find('div', {'class': 'mw-parser-output'})
    paragraphs = parser_output.find_all('p')
    #print(len(paragraphs))
    return paragraphs

def find_word_list_other(url: str) -> list[str]:
    html = call_url(url)
    soup = BeautifulSoup(html, features="lxml")
    try:
        jsAdd = soup.find('div', {'class': 'jsAdd'})
        paragraphs = jsAdd.find_all('li')
    except AttributeError:
        jsAdd = soup.find('div', {'class': 'mw-body-content mw-content-ltr'})
        paragraphs = jsAdd.find_all('li')
    
    return paragraphs
    

def get_words_word_list(L: list) -> list:
    target = []
    for i in range(4, 30):
        try:
            word_list = [
                item.get_text().lower() for item in L[i] if 
                '–' not in str(item.get_text()) and '-' not in str(item.get_text()) and 
                str(item.get_text()) != '\n' and len(item.get_text()) > 3
                ]
                
            target += word_list
        except IndexError:
            break
    return target

def __wiktionary(wiki_list: list[str]):
    download_wiktionary = {}
    for url in wiki_list:
        base = str(url.split('/')[-1])
        print(url)
        L = find_word_list(url)
        target = get_words_word_list(L)
        if len(target) > 1:
            download_wiktionary[base] = target
        else:
            L = find_word_list_other(url)
            target = [w.get_text() for w in L]
            if len(target) > 1:
                download_wiktionary[base] = target
    with open('data/download_wiktionary/download_wiktionary.json', 'w', encoding='utf-8') as f:
        json.dump(download_wiktionary, f) 

def download_wiktionary():
    __wiktionary(WIKI_LIST)

if __name__ == "__main__":
    download_wiktionary()
