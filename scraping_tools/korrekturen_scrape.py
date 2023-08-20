# to import scraper from same-level subdirectory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import json
import requests
import time

from bs4 import BeautifulSoup
from tqdm import tqdm
from scraping_tools.korrekturen_test import ortho_raw_to_readable

def run(path: str):
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    for type_ in ('spelling_error', 'orthography'):

        print(f"{type_.upper()}_LOOKUP")
        error_tuples = []

        if type_ == 'spelling_error':
            BASE_URL = "https://www.korrekturen.de/beliebte_fehler"
        if type_ == 'orthography':
            BASE_URL = "https://www.korrekturen.de/wortliste"

        print(BASE_URL)
        time.sleep(1)
        for _ in tqdm(alphabet):

            letter_url = f"{BASE_URL}/{_}/"

            for page in range(1, 6):
                if page == 1:
                    page_suffix = ""
                else:
                    page_suffix = f"index{page}.shtml"

                compiled_url = letter_url + page_suffix
                r = requests.get(compiled_url)
                r.encoding = 'utf-8'

                if "404" in r.text:
                    break

                soup = BeautifulSoup(r.text, features="lxml")

                body = soup.find('tbody')
                items = body.find_all('td')

                for i in range(0, 20*3, 3):
                    try:
                        spelling_error = str(items[i].get_text()).strip()
                        correct = str(items[i+1].get_text()).strip()

                        if type_ == 'orthography':
                            new = str(items[i+2].get_text()).strip()
                    except IndexError:
                        break

                    if type_ == 'spelling_error':
                        error_tuples.append((spelling_error, correct))
                    if type_ == 'orthography':
                        error_tuples.append((spelling_error, correct, new))

        print(f"crawled {len(error_tuples)} items.")
        error_json_output = {'errors': error_tuples}

        if type_ == 'spelling_error':
            file_name = f'{path}/annotation/error_tuples.json'
            with open(file_name, 'w') as f:
                json.dump(error_json_output, f)
        if type_ == 'orthography':
            ortho_raw_to_readable(error_json_output, path)

        
if __name__ == "__main__":
    run()

