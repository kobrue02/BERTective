import json
import requests
import trafilatura
import time
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

HEADERS = {'User-Agent': 'Mozilla/5.0 (iPad; CPU OS 12_2 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148'}

def achse_des_guten(url: str) -> str:

    html = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(html.text, features="lxml")

    content = __achgut_maintext(soup)
    author = __achgut_author(soup)

    return author, content

def __achgut_maintext(soup: BeautifulSoup) -> str:
    mainhtml = soup.find("div", {"id": "article_maincontent"})
    try:
        return trafilatura.html2txt(str(mainhtml))
    except AttributeError:
        return ""
    
def __achgut_author(soup: BeautifulSoup) -> str:
    mainhtml = soup.find("div", {"class": "teaser_text_meta"})
    raw = trafilatura.extract(str(mainhtml))
    try:
        author = raw.split(" / ")[0]
    except AttributeError:
        author = ""
    return author

def get_age_and_sex(url: str):
    author, content = achse_des_guten(url)

    with open("data/achse/authors.json", "r") as f:
        data = json.load(f)

    try:
        age = data["authors"][author]["age"]
        sex = data["authors"][author]["sex"]

    except KeyError:
        age, sex = None, None
        
    return age, sex, content

def get_all_articles():
    with open("data/achse/achse_des_guten.json", "w") as f:
            json.dump({'urls': []}, f)

    base = "https://www.achgut.com/P"

    for i in range(10, 65999, 13):

        urls_list = []

        source_url = base + str(i)
        print(source_url)

        html = requests.get(source_url, headers=HEADERS)
        soup = BeautifulSoup(html.text, features="lxml")

        urls = soup.find_all("div", {"class": "teaser_blog_text"})[:-1]
        
        for url in urls:
            item = url.find("h3")
            obj = item.find('a')
            link = obj.get('href')
            print(link)
            urls_list.append(link)

        time.sleep(0.5)

        with open("data/achse/achse_des_guten.json", "r+") as f:
            url_dict = json.load(f)
            url_dict['urls'] += urls_list
            f.seek(0)
            json.dump(url_dict, f)

def run():

    with open("data/achse/achse_des_guten.json", "r", encoding="utf-8") as f:
        urls = json.load(f)
    
    achse_dict = {}
    i = 0
    for url in tqdm(urls["urls"]):
        age, sex, content = get_age_and_sex(url)
        if any([age, sex]):
            achse_dict[f"txt {i}"] = {"content": content, "age": age, "sex": sex}
            i += 1

    achse_df = pd.DataFrame.from_dict(achse_dict).transpose()
    achse_df.reset_index(drop=True, inplace=True)
    print(achse_df.head())
    
    achse_df.to_excel("achse_des_guten_annotated_items.xlsx", index=False)

if __name__ == "__main__":

    run()