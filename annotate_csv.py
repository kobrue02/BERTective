"""
Mit diesem Skript können mithilfe von Regex-Patterns automatisch Alters- und Geschlechtsangaben
aus Reddit-Posts ausgelesen und annotiert werden.
Es wird ein pandas Dataframe zurückgegeben, welches in anderen Skripts verwendet, oder
als csv gespeichert werden kann.
"""
import json
import re
import pandas as pd
import os
from tqdm import tqdm
from models.zdl_vector_model import AREAL_DICT


def get_age_from_str(input_str: str) -> int:
    """ finds double-digit number within string """
    search = re.search(r"[1-9][0-9]", input_str)
    age = search.group(0)

    return int(age)

def get_gender_from_str(input_str: str) -> str:
    """ finds gender tag in string """
    search = re.search(r"(m|M|f|F)", input_str)
    if search:
        if search.group(0).upper() == "M":
            return "male"
        else:
            return "female"
    else:
        return "N/A"
    
def __clean_text(content: str, age) -> str:

    for put in [f"M{age}", f"{age}M", f"m{age}", f"{age}m", f"f{age}", f"{age}f", f"F{age}", f"{age}F", f"{age}"]:
        content = content.replace(put, "[REDACTED]")

    return content

def find_matches(data: dict, locale: str) -> list[dict]:

    """ finds gender or age tags within a reddit post and returns them 
    in an annotated dictionary """

    dict_of_texts_with_matches = {}

    for n, comment in enumerate(data["data"]):

        selftext: str = comment["selftext"]

        if selftext == "[removed]":
            continue

        m1 = re.finditer(r"(mir|ich) (\(|\[)(M|m|F|f)[1-9][0-9](M|m|F|f)?(\)|\])", selftext.lower())
        m2 = re.finditer(r"(bin) [1-9][0-9](M|m|F|f)?", selftext.lower())
        m3 = re.finditer(r"(\(|\[)[1-9][0-9], (m|w|f)(\)|\])", selftext.lower())

        result = [x.group() for x in m1] + [x.group() for x in m2] + [x.group() for x in m3] 
        result_as_str = "".join(result)
        if result:
            dict_of_texts_with_matches[f"item_{n}"] = {"match": result_as_str, "content": selftext}

    for item in dict_of_texts_with_matches.keys():
        row = dict_of_texts_with_matches[item]
        age = get_age_from_str(row["match"])
        gender = get_gender_from_str(row["match"])
        dict_of_texts_with_matches[item]["age"] = age
        dict_of_texts_with_matches[item]["sex"] = gender
        dict_of_texts_with_matches[item]["regiolect"] = locale

        content = dict_of_texts_with_matches[item]["content"]
        content = __clean_text(content, age)

        dict_of_texts_with_matches[item]["content"] = content

    return dict_of_texts_with_matches

def annotate_to_dataframe(path: str, locale: str) -> pd.DataFrame:
    with open(path, "r") as f:
        data = json.load(f)

    matches = find_matches(data, locale)

    text_data = pd.DataFrame.from_dict(matches).transpose()
    text_data.reset_index(drop=True, inplace=True)

    return text_data

if __name__ == "__main__":

    datasets = []
    path = "test"
    for subdir in ["dating"]: #, "education", "profession", "locales"]:
        print("annotating data from r/{}.".format(subdir))
        directory_in_str = f"{path}/reddit/{subdir}"
        directory = os.fsencode(directory_in_str)
        for file in tqdm(os.listdir(directory)):
            filename = os.fsdecode(file)
            if filename.endswith(".json"): 

                if subdir == "locales":
                    r = str(filename.split(".")[0])
                    for key in AREAL_DICT.keys():
                        if r in AREAL_DICT[key]:
                            locale = key
                else:
                    locale = ""
                text_data = annotate_to_dataframe(f"{directory_in_str}/{filename}", locale)
                datasets.append(text_data)


    data: pd.DataFrame = pd.concat(datasets)
    print(data.head())
    print(len(data.index))
    print(list(set(data.sex.tolist())))

    data.to_parquet('test/reddit/annotated_posts_2.parquet')