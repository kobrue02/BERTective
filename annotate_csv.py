import json
import re
import pandas as pd


def get_age_from_str(input_str: str) -> int:

    search = re.search(r"[1-9][0-9]", input_str)
    age = search.group(0)

    return int(age)

def get_gender_from_str(input_str: str) -> str:

    search = re.search(r"(m|M|f|F)", input_str)

    if search:
        return search.group(0).upper()

    else:
        return "N/A"


def find_matches(data: dict) -> list[dict]:

    good_list = {}

    for n, comment in enumerate(data["data"]):

        selftext = comment["selftext"]

        if selftext == "[removed]":
            continue

        m1 = re.finditer(r"(mir|ich) (\(|\[)(M|m|F|f)[1-9][0-9](\)|\])", selftext.lower())
        m2 = re.finditer(r"(bin) [1-9][0-9]", selftext.lower())
        m3 = re.finditer(r"(\(|\[)[1-9][0-9], (m|w|f)(\)|\])", selftext.lower())

        result = [x.group() for x in m1] + [x.group() for x in m2] + [x.group() for x in m3] 
        
        if result:
            good_list[f"item_{n}"] = {"match": "".join(result), "content": selftext}


    for item in good_list.keys():
        row = good_list[item]
        age = get_age_from_str(row["match"])
        gender = get_gender_from_str(row["match"])
        good_list[item]["age"] = age
        good_list[item]["sex"] = gender

    return good_list

def annotate_to_dataframe(path: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    matches = find_matches(data)

    text_data = pd.DataFrame.from_dict(matches).transpose()
    text_data.reset_index(drop=True, inplace=True)

    return text_data

if __name__ == "__main__":

    text_data = annotate_to_dataframe("data/reddit/Ratschlag.json")
    print(text_data.head())