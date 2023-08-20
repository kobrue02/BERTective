import datetime
import requests
import json
from tqdm import tqdm

# to import scraper from same-level subdirectory
import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from models.zdl_vector_model import AREAL_DICT

def pushshift_request(subreddit: str, limit: str, before: str = "0", after: str = "0"):

    url = __pushshift_url(subreddit, limit, before, after)
    r = requests.get(url)
    json_data = r.json()

    return json_data

def __pushshift_url(subreddit: str, limit: str, before: str, after: str) -> str:

    if before != "0":
        year, month, day = __to_ymd(before)
        epoch_before = __to_epoch(year, month, day)
    else:
        epoch_before = "0"

    if after != "0":
        year, month, day = __to_ymd(after)
        epoch_after = __to_epoch(year, month, day)
    else:
        epoch_after = "0"

    url = f"https://api.pullpush.io/reddit/submission/search?html_decode=true&before={epoch_before}&after={epoch_after}&subreddit={subreddit}&size={limit}"
    return url

def __to_ymd(date: str) -> tuple:

    year = int(date.split(".")[2])
    month = int(date.split(".")[1])
    day = int(date.split(".")[0])

    return year, month, day

def __to_json_file(data: dict, filename: str, path: str):

    try:
        with open(f"{path}/reddit/locales/{filename}.json", "r+", encoding="UTF-8") as f:
            old_data = json.load(f)
    except FileNotFoundError:
        old_data = {"data": []}
        
    with open(f"{path}/reddit/locales/{filename}.json", "w", encoding="UTF-8") as f:
        new_data = {"data": [item for item in old_data["data"] + [item for item in data["data"]]]}
        json.dump(new_data, f)

def __to_epoch(year: int, month: int, day: int) -> str:

    epoch = int(datetime.datetime(year, month, day, 0, 0).timestamp())
    return str(epoch)

def __filename(obj: dict) -> str:

    subreddit = obj["data"][0]["subreddit"]
    return subreddit

def __download_subreddit_batch(subreddit: str, path: str):
    last_data = {}
    year = 2023
    while year > 2016:

        if year == 2023:
            month = 8
        else:
            month = 12
                
        month = int(month)
        while month > 0:
                    
            if month < 10:
                month = str(f"0{month}")

            tqdm.write(f"28.{str(month)}.{str(year)}")
            data = pushshift_request(subreddit, "100", before=f"28.{str(month)}.{str(year)}")
            if data == last_data:
                tqdm.write('no new data.')
                pass
            else:
                last_data = data
                tqdm.write('crawled new batch.')
                try:
                    file = __filename(data)
                    __to_json_file(data, file, path)
                except:
                    pass
            month = int(month) - 1
        year -= 1

def locale_reddits(path: str):
    for val_list in tqdm(AREAL_DICT.values()):
        for locale in val_list:
            __download_subreddit_batch(locale, path)
            

if __name__ == "__main__":
    
    locale_reddits('test')
