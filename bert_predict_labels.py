#from transformers import BertTokenizerFast
import jsonlines
import pandas as pd
from pprint import pprint

#tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

def jsonline_to_row(label: str, prodigy_item: dict) -> str:

    return_row = ""
    label_span = 0

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

path = "data/annotation/sentence_level_annotations.jsonl"
with jsonlines.open(path, mode='r') as reader:
    objects = [item for item in reader]

for object in objects:
    try:
        if len(object["spans"]) > 1:
            pprint(object)
            print('-' * 64)
            print(jsonline_to_row('Interpkt.', object))
    except KeyError:
        pprint(object)
        continue





training_dataset = pd.DataFrame()


