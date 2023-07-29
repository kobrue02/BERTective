import jsonlines
import pandas as pd
import nltk
import os

from annotate_csv import annotate_to_dataframe
from tqdm import tqdm

def get_filenames() -> list[str]:

    files = []

    for source in ("reddit", "achse"):    
        for file in os.scandir(f"data/{source}/"):
            filename = file.path
            if filename.endswith('.xlsx') or filename.endswith('.json'): 
                files.append(filename)
            else:
                continue
    return files

def dataframe_to_sentences(raw_data: pd.DataFrame()) -> list:
    text_items = raw_data["content"].tolist()

    sentences = []

    for source in text_items:
        sents_in_source = nltk.sent_tokenize(source)

        sentences += sents_in_source

    return sentences

def dataframe_to_words(raw_data: pd.DataFrame()) -> list:
    text_items = raw_data["content"].tolist()

    words = []

    for source in text_items:
        words_in_source = nltk.word_tokenize(source)

        words += words_in_source

    return words


def list_to_jsonl(sentences: list, source: str, target_path: str) -> None:
    jsonl_objects = []
    for item in sentences:
        jsonl_obj = {'text': item, 'meta': {'source': source}}
        jsonl_objects.append(jsonl_obj)

    with jsonlines.open(f'{target_path}.jsonl', mode='a') as writer:
        for item in jsonl_objects:
            writer.write(item)

def unique(items: list) -> list:
    return list(set(items))

def __dedupe(path: str):
    unique_items = []
    with jsonlines.open(f'{path}.jsonl', mode='r') as reader:
        objects = [item for item in reader]
        for item in objects:
            if item not in unique_items:
                unique_items.append(item)
    with jsonlines.open(f'{path}_dedupe.jsonl', mode='a') as writer:
        for item in unique_items:
            writer.write(item)

def main(level: str):
    files = get_filenames()

    for file in tqdm(files):
        if file.endswith('.xlsx'):
            data = pd.read_excel(file)
            source_name = "Achse des Guten"
        elif file.endswith('.json'):
            try:
                data = annotate_to_dataframe(file)
            except KeyError:
                continue
            source_name = f"r/{file.split('/')[-1].split('.')[0]}"
        else:
            continue
        if not data.empty:
            if level in 'sentences':
                sentences = dataframe_to_sentences(data)
                list_to_jsonl(sentences, source_name, 'sentences')
            elif level in 'words':
                words = unique(dataframe_to_words(data))
                list_to_jsonl(words, source_name, 'words')
            else:
                raise UnboundLocalError('level has to be sentences or words')
        else:
            continue



if __name__ == "__main__":

    main('words')
    __dedupe('sentences')