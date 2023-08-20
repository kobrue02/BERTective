from corpus import DataCorpus, DataObject
from crawl_all_datasets import download_data

import argparse
import os
import pandas as pd


if __name__ == "__main__":

    PATH = "test"

    os.makedirs(PATH, exist_ok=True)
    os.makedirs(f'{PATH}/achse', exist_ok=True)
    os.makedirs(f'{PATH}/annotation', exist_ok=True)
    os.makedirs(f'{PATH}/reddit', exist_ok=True)
    os.makedirs(f'{PATH}/reddit/locales', exist_ok=True)

    download_data(['achse', 'ortho', 'reddit_locales'], "test")

    data = DataCorpus()

    achse = pd.read_parquet(f'{PATH}/achse/achse_des_guten_annotated_items.parquet')

    for item in zip(achse.content, achse.age, achse.sex):
        obj = DataObject(
            text = item[0],
            author_age=item[1],
            author_gender=item[2],
            source="ACHGUT"
        )

        data.add_item(obj)

    print(data.as_dataframe().head())
