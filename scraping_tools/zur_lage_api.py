import json
import requests
import trafilatura
import time
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}

with open("data/zur_lage/zur_lage_authors.json") as f:
    authors = json.load(f)

def zur_lage_api_call(url: str) -> tuple[str]:

    html = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(html.text, features="lxml")
    mainhtml = soup.find("div", {"class": "entry-content clear"})
    
    try:
        text = trafilatura.html2txt(str(mainhtml))
    
    except AttributeError:
        text = ""

    author_html = soup.find_all("h4")
    author = str(author_html[0].get_text())

    try:
        author_age = authors["authors"][author]["age"]
        author_sex = authors["authors"][author]["sex"]
    except KeyError:
        return None

    result = {"text": text,
              "author_name": author,
              "author_age": author_age,
              "author_sex": author_sex}
    return result


def iterate_over_articles(url: str) -> list:
    html = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(html.text, features="lxml")
    article_list = []

    for item in soup.find_all('a', {'class': 'ast-button'}):
        article_list.append(item.get('href'))
    
    return article_list

def iterate_over_pages(url: str) -> list:

    article_list = []

    #page 0
    list_1 = iterate_over_articles(url)
    article_list += list_1

    url += 'page/'

    for n in tqdm(range(2,56)):
        page_url = url + f'{n}/'
        curr_list = iterate_over_articles(page_url)
        article_list += curr_list
    
    return article_list

    
    


if __name__ == "__main__":
    #print(zur_lage_api_call("https://zur-lage.com/2022/08/tag-der-begegnung-mit-der-bundeswehr/"))
    articles = iterate_over_pages('https://zur-lage.com/category/zur-lage/')

    article_dicts = []
    for article in tqdm(articles):
        content_tup = zur_lage_api_call(article)
        #tqdm.write(content_tup)
        if content_tup is not None:
            print(content_tup)
            article_dicts.append(content_tup)

    output_json = {"items": article_dicts}
    with open('data/zur_lage/zur_lage_text_items.json', "w") as f:
        json.dump(output_json, f)