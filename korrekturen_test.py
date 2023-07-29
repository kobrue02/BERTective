import jsonlines
import json
import nltk
import time
import re

from tqdm import tqdm

with open('rechtschreibung.json', 'r') as f:

    # tabelle von korrekturen.de wird "roh" eingelesen
    raw_triplets = json.load(f)
    all_triples = raw_triplets["errors"]

    # tabelle wird in spalten gesplittet
    ancient_orthography = [str(item[0]) for item in all_triples]
    revolutionized_orthography = [str(item[1]) for item in all_triples]
    modern_orthography = [str(item[2]) for item in all_triples]

    print(len(ancient_orthography))


def clean_list(L: list) -> list:
    """
    Säubert eine Spalte aus der Rechtschreibungstabelle von korrekturen.de
    """
    return_list = []
    for i in range(len(L)):
        item = L[i]
        fixed_item = item.split("\n")[0]
        fixed_item = fixed_item.split(";")[0]

        # wenn es klammern gibt, werden diese mit inhalt entfernt
        try:
            p_content = re.search(r'\((.*?)\)', fixed_item).group(1)
        except AttributeError:
            p_content = ""
        fixed_item = fixed_item.replace(f"({p_content})", '')
        fixed_item = fixed_item.split("(")[0]
        fixed_item = fixed_item.split("/")[0].strip()
        
        # Erläuterungen werden ebenfalls entfernt
        if "Bedeutung" in fixed_item:
            fixed_item = fixed_item.replace("in übertragener Bedeutung", "")
            fixed_item = fixed_item.replace("In übertragener Bedeutung", "")
            fixed_item = fixed_item.replace("in wörtlicher Bedeutung", "")
            fixed_item = fixed_item.replace("In wörtlicher Bedeutung", "")
        if ":" in fixed_item:
            fixed_item = fixed_item.replace(", auch:", "")
            print(fixed_item)
        return_list.append(fixed_item)
    return return_list

ortho_dict = {"orthographies": {}}
for ortho in (ancient_orthography, revolutionized_orthography, modern_orthography):
    print("-"*64)
    time.sleep(1)
    clean_orthography = clean_list(ortho)
    if ortho is ancient_orthography:
        var_name = "ancient"
    elif ortho is revolutionized_orthography:
        var_name = "revolutionized"
    elif ortho is modern_orthography:
        var_name = "modern"
    else:
        raise KeyError
    ortho_dict["orthographies"][var_name] = clean_orthography

with open(f"data/annotation/orthography.json", "w") as f:
    json.dump(ortho_dict, f)
 
exit()
sentences = []
with jsonlines.open('data/annotation/sentences_dedupe.jsonl', mode='r') as reader:
    for line in reader:
        sentences.append(line['text'])

for sent in tqdm(sentences):
    sent_tokens = nltk.word_tokenize(sent)
    for error in error_tuples:
        if error in sent_tokens:
            tqdm.write(f"found '{error}' in {sent}.")
            time.sleep(1)