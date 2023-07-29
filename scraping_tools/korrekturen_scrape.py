import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


error_tuples = []
alphabet = "abcdefghijklmnopqrstuvwxyz"

#base_url = "https://www.korrekturen.de/beliebte_fehler"
base_url = "https://www.korrekturen.de/wortliste"

for _ in tqdm(alphabet):

    letter_url = f"{base_url}/{_}/"

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
                error = str(items[i].get_text()).strip()
                correct = str(items[i+1].get_text()).strip()
                new = str(items[i+2].get_text()).strip()
            except IndexError:
                break
            tqdm.write(f"{error}: {correct}")
            error_tuples.append((error, correct, new))

print(len(error_tuples))
error_json_output = {'errors': error_tuples}

with open('rechtschreibung.json', 'w') as f:
    json.dump(error_json_output, f)

