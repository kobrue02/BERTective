import json
import requests
import trafilatura
import time

from bs4 import BeautifulSoup
from tqdm import tqdm

def __fix_url(url: str) -> str:

    delete = url.split("/")
    delete = delete[:-1]
    merged = "/".join(delete)

    return merged + "/chap"

url = 'https://www.projekt-gutenberg.org/info/texte/allworka.html'
r = requests.get(url)
r.encoding = 'UTF-8'
soup = BeautifulSoup(r.text, features="lxml")

book_links = []
links_on_main = soup.find_all("dd")

author_book_list = []

for link in links_on_main:
    anchor = link.find("a")
    try: 
        url = anchor.get("href")
    except:
        continue
    book_links.append(url)


def __get_all_chapters(url: str) -> list:
    abl = []
    url = str(url).replace('../../', 'https://www.projekt-gutenberg.org/')
    # get author name

    got_response = False
    while not got_response:
        try:
            re = requests.get(url)
            got_response = True
        except:
            tqdm.write('retrying.')
            time.sleep(1)

    soup = BeautifulSoup(re.text, features="lxml")
    author = soup.find('meta', attrs={'name': 'author'})
    book = soup.find('meta', attrs={'name': 'title'})

    try:
        author_name = author.get("content")
    except:
        return

    try:
        book_name = book.get("content")
    except:
        return

    url = __fix_url(url)
    for i in range(1,101):
        tqdm.write(f'downloaded chapter {i} of {book_name}.')
        chapter = url + f"{i:03d}" + ".html"
        r = requests.get(chapter)
        r.encoding = 'UTF-8'
        text_content = trafilatura.extract(r.text)
        error_phrase = """10.000 Werke lokal lesen: Gutenberg-DE Edition 15 auf USB. Information und Bestellung in unserem Shop"""
        if error_phrase in text_content:
            break
  
        book_dict = {
            'author_name': author_name,
            'author_age': '',
            'author_gender': '',
            'book_name': book_name,
            'text': text_content
            }
      
        abl.append(book_dict)
    return abl


for url in tqdm(book_links[2:]):
    try:
        abl = __get_all_chapters(url)
        author_book_list += abl
    except: 
        time.sleep(1)
        continue

output_json = {'texts': author_book_list}
with open('data.json', 'w') as f:
    json.dump(output_json, f)