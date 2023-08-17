#from transformers import BertTokenizerFast
import jsonlines
import pandas as pd
from pprint import pprint

#tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased')

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

path = "data/annotation/sentence_level_annotations.jsonl"
all_labels = ['wOrder', 'wFehlt', 'Interpkt.', 'Rechtschr.', 'ugs.', 'Angliz.', 'geh.', 'emo.', 'wRedund', 'Jarg.']
label_item_lists = {}

with jsonlines.open(path, mode='r') as reader:
    objects = [item for item in reader]

for label in all_labels:
    label_item_lists[label] = [jsonline_to_row(label, obj) for obj in objects]
label_item_lists['text'] = [obj['text'] for obj in objects]
    
training_dataset = pd.DataFrame(label_item_lists)
print(training_dataset.head())


