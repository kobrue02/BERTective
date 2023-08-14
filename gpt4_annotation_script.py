import ast
import json
import openai
import time
secret_key = ""    # to be filled locally

openai.api_key = secret_key

def __response(author: str, book: str):
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": """
            You will be provided with names of authors and books they have written, 
            and your task is to provide the age of the author at the time of writing the book,
            as well as their degree of education and sex.
            You should provide the answer as a python dictionary: {'age': xx, 'sex': m/f, 'degree of education': xyz}
            If you know the author, but don't know the book, return their estimated age during the peak of their career.
            If you dont know the author at all, just return None for each key. """
        },
        {
        "role": "user",
        "content": f"{author}: {book}"
        }
    ],
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response["choices"][0]["message"]["content"]


if __name__ == "__main__":

    author_dict = {}
    with open('data/gutenberg/data.json', "r", encoding='utf-8') as f:
        data = json.load(f)

    author = ""
    book = ""
    for item in data["texts"]:
        if item["author_name"] == author:
            continue
        author = item["author_name"]
        book = item["book_name"]
        try:
            raw_response = __response(author, book)
        except (openai.error.ServiceUnavailableError, openai.error.APIConnectionError):
            time.sleep(5)
            continue
        try:
            response_dict = ast.literal_eval(raw_response)
        except SyntaxError:
            print(raw_response)
            continue
        print(f"{author}: \n{response_dict}")
        author_dict[author] = response_dict

with open('output_json.json', 'w', encoding='utf-8') as f:
    json.dump(author_dict, f) 
