import json
from pprint import pprint


def align_dicts(data: dict, author_dict: dict) -> dict:
    full_dict = {"books": []}
    for data_item in data['texts']:
        try:
            y = next(item for item in author_dict['books'] if item["book_name"] == data_item['book_name'])
        except StopIteration:
            break
        temp = dict(y)
        temp.update({'text': data_item["text"]})
        full_dict['books'].append(temp)
    return full_dict

if __name__ == "__main__":

    with open('data/gutenberg/data.json', 'r') as f:
        data = json.load(f)

    with open('data/gutenberg/author_dict.json', 'r') as f:
        author_dict = json.load(f)

    authors = align_dicts(data, author_dict)

    