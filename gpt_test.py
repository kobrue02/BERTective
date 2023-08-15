"""
Ein Test, ob GPT-4 sich zum automatischen Annotieren demografischer Eigenschaften bekannter 
Autoren eignet. Bei mehrmaligem Ausführen des Skripts erhält man verschiedene Antworten.
Es eignet sich also nicht.
"""

import openai
import time

secret_key = ""  # to be filled locally
openai.api_key = secret_key

def __response():
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {
        "role": "system",
        "content": """You are a helpful assistant with large real world knowledge."""
        },
        {
        "role": "user",
        "content": f"""How old was Edmond About when he wrote "der Bergkönig"""""
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
    print(__response())