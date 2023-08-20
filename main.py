from corpus import DataCorpus, DataObject
from crawl_all_datasets import download_data

import pandas as pd

if __name__ == "__main__":

    download_data(['ortho', 'achse'], "test/")

    data = DataCorpus()

    achse = pd.read_parquet('test/achse_des_guten_annotated_items.parquet')

    for item in zip(achse.content, achse.age, achse.sex):
        obj = DataObject(
            text = item[0],
            author_age=item[1],
            author_gender=item[2],
            source="ACHGUT"
        )

        data.add_item(obj)

    print(data.as_dataframe().head())
