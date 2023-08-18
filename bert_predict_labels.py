"""
Dieses Skript liest eine JSONL-Datei mit annotierten Sätzen.
Die Annotationen werden in ein für BERT verwertbares Format umgewandelt.
Anschließend kann BERT auf den Annotationen trainiert werden, um diese anschließend selber vorherzusagen.
https://towardsdatascience.com/named-entity-recognition-with-bert-in-pytorch-a454405e0b6a
wurde als Inspiration verwendet.
"""

import jsonlines
import pandas as pd
import numpy as np
from pprint import pprint
from models.bert_model import BertModel, train_loop, evaluate

def jsonline_to_row(label: str, prodigy_item: dict) -> str:

    return_row = ""
    label_span = 0

    if 'spans' not in list(prodigy_item.keys()):
        return str("O " * len(prodigy_item['tokens']))[:-1]
    if len(prodigy_item['spans']) == 0:
        return str("O " * len(prodigy_item['tokens']))[:-1]
    
    for span in prodigy_item['spans']:
        if span['label'] == label:
            label_span = range(span['token_start'], span['token_end'] + 1)
    if label_span == 0:
        return str("O " * len(prodigy_item['tokens']))[:-1]
    for token in prodigy_item['tokens']:
        if token['id'] not in label_span:
            return_row += "O "
        else:
            return_row += "X "
    return return_row[:-1]


def generate_dataset(path: str) -> pd.DataFrame:
    all_labels = ['wOrder', 'wFehlt', 'Interpkt.', 'Rechtschr.', 'ugs.', 'Angliz.', 'geh.', 'emo.', 'wRedund', 'Jarg.']
    label_item_lists = {}

    with jsonlines.open(path, mode='r') as reader:
        objects = [item for item in reader]

    for label in all_labels:
        label_item_lists[label] = [jsonline_to_row(label, obj) for obj in objects]
    label_item_lists['text'] = [obj['text'] for obj in objects]
        
    training_dataset = pd.DataFrame(label_item_lists)
    return training_dataset
    
if __name__ == "__main__":
    model = BertModel()
    path = "data/annotation/sentence_level_annotations.jsonl"
    df = generate_dataset(path)
    for label in ['wOrder', 'wFehlt', 'Interpkt.', 'Rechtschr.', 'ugs.', 'Angliz.', 'geh.', 'emo.', 'wRedund', 'Jarg.']:
        print(label)
        temp_df = df[["text", label]]
        df_train, df_val, df_test = np.split(temp_df.sample(frac=1, random_state=42),
                                [int(.8 * len(temp_df)), int(.9 * len(temp_df))])
        train_loop(model, df_train, df_val)
        evaluate(model, df_test)
        print("-"*64)
