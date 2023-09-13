import spacy
import morfessor


nlp = spacy.load("de_core_news_sm")

try:  # TODO: model
    model = morfessor.MorfessorIO().read_binary_model_file('/path/to/german_morfessor_model.bin')
except FileNotFoundError:
    model = None

VOWELS = 'aeiouäöüáéíóúàèìòùâêîôûAEIOUÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ'

CONSONANTS = 'bcdfghjklmnpqrstvwxyzßBCDFGHJKLMNPQRSTVWXYZẞ'

NON_CAPITALIZED = 'abcdefghijklmnopqrstuvwxyzäöüáéíóúàèìòùâêîôûß'

CAPITALIZED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛẞ'


class Statistext:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.doc = nlp(raw_text)
        self.char_word_ratio = self.__calculate_char_word_ratio()
        self.word_sent_ratio = self.__calculate_word_sent_ratio()
        self.morph_word_ratio = self.__calculate_morph_word_ratio()
        self.vowel_cons_ratio = self.__calculate_vowel_cons_ratio()
        self.cap_nocap_ratio = self.__calculate_cap_nocap_ratio()

    def __calculate_char_word_ratio(self) -> float:
        """
        Calculate the average count of characters per word
        :return: text’s average word length
        """
        total_words_length = 0
        word_count = 0
        for token in self.doc:
            if not token.is_space and not token.is_punct and not token.is_digit:
                total_words_length += len(token.text)
                word_count += 1
        if word_count > 0:
            return total_words_length/word_count
        else:
            return word_count

    def __calculate_word_sent_ratio(self) -> float:
        """
        Calculate the average count of words per sentence
        :return: text’s average word count per sentence
        """
        total_sent_count = 0
        total_word_count = 0
        for sentence in self.doc.sents:
            total_sent_count += 1
            for token in sentence:
                if not token.is_space and not token.is_punct:
                    total_word_count += 1
        if total_sent_count > 0:
            return total_word_count/total_sent_count
        else:
            return total_sent_count

    def __calculate_morph_word_ratio(self) -> float:
        """
        Calculate the average count of morphemes per word
        :return: text’s average morpheme count per word
        """
        if not model:  # TODO: model
            return float('inf')
        total_word_count = 0
        total_morph_count = 0
        for word in self.doc:
            total_word_count += 1
            total_morph_count += len(model.viterbi_segment(word))
        if total_word_count > 0:
            return total_morph_count/total_word_count
        else:
            return total_word_count

    def __calculate_vowel_cons_ratio(self) -> float:
        """
        Calculate the vowel-consonant ratio
        :return: text’s vowel-consonant ratio
        """
        vowel_count = 0
        consonant_count = 0
        for char in self.raw_text:
            if char in VOWELS:
                vowel_count += 1
            elif char in CONSONANTS:
                consonant_count += 1
        if consonant_count > 0:
            return vowel_count/consonant_count
        else:
            return consonant_count

    def __calculate_cap_nocap_ratio(self, count_remaining=False) -> float:
        """
        Calculate the ratio of capitalized to non-capitalized words
        :param count_remaining: for debugging, additionally print number of words that flew under the radar
        :return: text’s word capitalization rate
        """
        cap_count = 0
        nocap_count = 0
        neither_count = 0
        for token in self.doc:
            if not token.is_space and not token.is_punct and not token.is_digit:
                if str(token)[0] in NON_CAPITALIZED:
                    nocap_count += 1
                elif str(token)[0] in CAPITALIZED:
                    cap_count += 1
                else:
                    neither_count += 1
        if count_remaining:
            print(f"number of initial letters neither in NON_CAPITALIZED nor in CAPITALIZED: {neither_count}")
        if nocap_count > 0:
            return cap_count/nocap_count
        else:
            return nocap_count
