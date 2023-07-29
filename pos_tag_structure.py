import spacy

nlp = spacy.load('de_core_news_md')

error_phrase_patterns = [
    '[NOUN] [PUNCT] [SCONJ]'
]

def __string_to_pos_pattern(sentence: str) -> list:

    doc = nlp(sentence)
    pattern = ""

    for token in doc:
        pattern += f"[{token.pos_}] "

    return pattern[:-1]

def __check_for_error_span(sentence: str, error_patterns: list[str]) -> bool:

    sentence = __string_to_pos_pattern(sentence)

    for span in error_patterns:
        if span in sentence:
            return True
        
    return False

if __name__ == "__main__":

    with open('test_sentences.txt', 'r', encoding='utf-8') as f:
        lines = [line.replace('\n','') for line in f.readlines()]

    for input_sentence in lines:
        contains_error = __check_for_error_span(input_sentence, error_phrase_patterns)
        if contains_error:
            print(f'"{input_sentence}" contains an invalid POS-pattern.')
        else:
            print(f'"{input_sentence}" does not contain an invalid POS-pattern.')