import requests
import time
import json

from bs4 import BeautifulSoup
from pprint import pprint

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

WIKI_LIST = [
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Abkürzungen_im_Internet',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Anglizismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Arabismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Aufwertung',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Binomiale',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Biologie/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Chemie/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Chinesische_Wörter_in_der_deutschen_Sprache',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/DDR-Sprache',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Gallizismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Geowissenschaften',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Geschichte',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Hispanismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Japanische_Wörter_in_der_deutschen_Sprache',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Jägersprache',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Mathematik/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Medizin/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Militär/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Musik/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Netzjargon',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Ornithologie/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Pharmazie',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Philosophie',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Physik/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Politik',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Rechtswissenschaften',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Religion',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Romanismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Slawismen',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Weinbau',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Wirtschaft',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Griechische_Präfixe',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Griechische_Suffixe',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Informationstechnik',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Lateinische_Präfixe',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Lateinische_Suffixe',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Linguistik/Fachwortliste',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Philatelie',
        'https://de.wiktionary.org/wiki/Verzeichnis:Deutsch/Sprichwörter'
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

def __wiktionary(wiki_list: list[str], PATH):
    wiktionary = {}
    for url in wiki_list:
        base = str(url.split('/')[-1])
        L = find_word_list(url)
        target = get_words_word_list(L)
        if len(target) > 1:
            wiktionary[base] = target
        else:
            L = find_word_list_other(url)
            target = [w.get_text() for w in L]
            if len(target) > 1:
                wiktionary[base] = target
    with open(f'{PATH}/wiktionary/wiktionary.json', 'w', encoding='utf-8') as f:
        json.dump(wiktionary, f) 

def download_wiktionary(PATH):
    __wiktionary(WIKI_LIST, PATH)

if __name__ == "__main__":
    download_wiktionary()
