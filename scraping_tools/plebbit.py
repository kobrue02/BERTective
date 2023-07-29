import datetime
import requests
import json


def pushshift_request(subreddit: str, limit: str, before: str = "0", after: str = "0"):

    url = __pushshift_url(subreddit, limit, before, after)
    r = requests.get(url)
    print(r)

    json_data = r.json()
    print(json_data)
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

def __to_json_file(data: dict, filename: str):

    with open(f"data/reddit/education/{filename}.json", "r+", encoding="UTF-8") as f:
        old_data = json.load(f)
    
    with open(f"data/reddit/education/{filename}.json", "w", encoding="UTF-8") as f:
        new_data = {"data": [item for item in old_data["data"] + [item for item in data["data"]]]}
        json.dump(new_data, f)

def __to_epoch(year: int, month: int, day: int) -> str:

    epoch = int(datetime.datetime(year, month, day, 0, 0).timestamp())
    return str(epoch)

def __filename(obj: dict) -> str:

    subreddit = obj["data"][0]["subreddit"]
    return subreddit

if __name__ == "__main__":
    
    data = pushshift_request("Azubis", "100", before="05.07.2023")
    print(data)
    file = __filename(data)
    __to_json_file(data, file)
