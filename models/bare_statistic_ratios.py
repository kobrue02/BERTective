import spacy
import emoji

nlp = spacy.load("de_core_news_sm")

VOWELS = 'aeiouäöüáéíóúàèìòùâêîôûAEIOUÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ'

CONSONANTS = 'bcdfghjklmnpqrstvwxyzßBCDFGHJKLMNPQRSTVWXYZẞ'

NON_CAPITALIZED = 'abcdefghijklmnopqrstuvwxyzäöüáéíóúàèìòùâêîôûß'

CAPITALIZED = 'ABCDEFGHIJKLMNOPQRSTUVWXYZÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛẞ'

with open("models/wikipedia_emoticons.txt", encoding='utf-8') as f:
    EMOTICONS = f.read().split("\n")
    """
    This file is a manually edited version of the lists at https://en.wikipedia.org/wiki/List_of_emoticons.
    TODO: verify copyright stuff
    FIXME: finally make list not contain funny unicode escape seqs anymore
    """

class Statistext:
    def __init__(self, raw_text):
        self.raw_text = raw_text
        self.doc = nlp(raw_text)
        self.characters_per_word = self.__calculate_characters_per_word()
        self.words_per_sentence = self.__calculate_words_per_sentence()
        # self.morphemes_per_sentence = "This function is currently not working and might be removed later. The chars-per-[POS] functions should achieve similar results. [possible future feature]"
        self.characters_per_noun = self.__calculate_characters_per_noun()
        self.characters_per_verb = self.__calculate_characters_per_verb()
        self.characters_per_adjective = self.__calculate_characters_per_adjective()
        self.characters_per_adverb = self.__calculate_characters_per_adverb()
        self.nouns_per_sentence = self.__calculate_nouns_per_sentence()
        self.verbs_per_sentence = self.__calculate_verbs_per_sentence()
        self.adjectives_per_sentence = self.__calculate_adjectives_per_sentence()
        self.adverbs_per_sentence = self.__calculate_adverbs_per_sentence()
        self.vowel_to_consonant_ratio = self.__calculate_vowel_to_consonant_ratio()
        self.capped_to_notcapped_ratio = self.__calculate_capped_to_notcapped_ratio()
        self.emoji_count = self.__calculate_emoji_count()
        self.emoticon_count = self.__calculate_emoticon_count()
        self.all_stats = {
            "characters_per_word": self.characters_per_word,
            "words_per_sentence": self.words_per_sentence,
            "characters_per_noun": self.characters_per_noun,
            "characters_per_verb": self.characters_per_verb,
            "characters_per_adjective": self.characters_per_adjective,
            "characters_per_adverb": self.characters_per_adverb,
            "nouns_per_sentence": self.nouns_per_sentence,
            "verbs_per_sentence": self.verbs_per_sentence,
            "adjectives_per_sentence": self.adjectives_per_sentence,
            "adverbs_per_sentence": self.adverbs_per_sentence,
            "vowel_to_consonant_ratio": self.vowel_to_consonant_ratio,
            "capped_to_notcapped_ratio": self.capped_to_notcapped_ratio,
            "emoji_count": self.emoji_count,
            "emoticon_count": self.emoticon_count,
        }


    def __calculate_characters_per_word(self) -> float:
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

    def __calculate_words_per_sentence(self) -> float:
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

    # def __calculate_morph_word_ratio(self) -> float:
    #     """
    #     Calculate the average count of morphemes per word
    #     :return: text’s average morpheme count per word
    #     """
    #     """
    #     FIXME
    #     currently needs model that can recognize morpheme boundaries, model not found nor trained atm – avg. noun length etc. might be a good alternative
    #     [this FIXME references same problem as one in class attribute line above]
    #     """
    #     try:
    #         total_word_count = 0
    #         total_morph_count = 0
    #         for word in self.doc:
    #             total_word_count += 1
    #             total_morph_count += len(model.viterbi_segment(word))
    #         if total_word_count > 0:
    #             return total_morph_count/total_word_count
    #         else:
    #             return total_word_count
    #     except NameError:
    #         return 420.69/420.69

    def __calculate_characters_per_noun(self, include_propn=False) -> float:
        """
        :param include_propn: If set to True, count POS tags NOUN and PROPN instead of NOUN only.
        :return: average length of nouns in the text
        """
        pos_tags = ["NOUN"]
        if include_propn:
            pos_tags.append("PROPN")
        total_nouns_length = 0
        noun_count = 0
        for token in self.doc:
            if token.pos_ in pos_tags:
                total_nouns_length += len(token.text)
                noun_count += 1
        if noun_count > 0:
            return total_nouns_length/noun_count
        else:
            return noun_count

    def __calculate_characters_per_verb(self, include_aux=False) -> float:
        """
        :param include_aux: If set to True, count POS tags VERB and AUX instead of VERB only.
        :return: average length of verbs in the text
        """
        pos_tags = ["VERB"]
        if include_aux:
            pos_tags.append("AUX")
        total_verbs_length = 0
        verb_count = 0
        for token in self.doc:
            if token.pos_ in pos_tags:
                total_verbs_length += len(token.text)
                verb_count += 1
        if verb_count > 0:
            return total_verbs_length/verb_count
        else:
            return verb_count

    def __calculate_characters_per_adjective(self) -> float:
        """
        :return: average length of adjectives in the text
        """
        pos_tags = ["ADJ"]
        total_adjectives_length = 0
        adjective_count = 0
        for token in self.doc:
            if token.pos_ in pos_tags:
                total_adjectives_length += len(token.text)
                adjective_count += 1
        if adjective_count > 0:
            return total_adjectives_length/adjective_count
        else:
            return adjective_count

    def __calculate_characters_per_adverb(self) -> float:
        """
        :return: average length of adverbs in the text
        """
        pos_tags = ["ADV"]
        total_adverbs_length = 0
        adverb_count = 0
        for token in self.doc:
            if token.pos_ in pos_tags:
                total_adverbs_length += len(token.text)
                adverb_count += 1
        if adverb_count > 0:
            return total_adverbs_length/adverb_count
        else:
            return adverb_count
        
    def __calculate_nouns_per_sentence(self, include_propn=False) -> float:
        """
        :param include_propn: If set to True, count POS tags NOUN and PROPN instead of NOUN only.
        :return: average number of nouns per sentence
        """
        pos_tags = ["NOUN"]
        if include_propn:
            pos_tags.append("PROPN")
        total_sent_count = 0
        total_noun_count = 0
        for sentence in self.doc.sents:
            total_sent_count += 1
            for token in sentence:
                if token.pos_ in pos_tags:
                    total_noun_count += 1
        if total_sent_count > 0:
            return total_noun_count/total_sent_count
        else:
            return total_sent_count

    def __calculate_verbs_per_sentence(self, include_aux=False) -> float:
        """
        :param include_aux: If set to True, count POS tags VERB and AUX instead of VERB only.
        :return: average number of verbs per sentence
        """
        pos_tags = ["VERB"]
        if include_aux:
            pos_tags.append("AUX")
        total_sent_count = 0
        total_verb_count = 0
        for sentence in self.doc.sents:
            total_sent_count += 1
            for token in sentence:
                if token.pos_ in pos_tags:
                    total_verb_count += 1
        if total_sent_count > 0:
            return total_verb_count/total_sent_count
        else:
            return total_sent_count
        
    def __calculate_adjectives_per_sentence(self) -> float:
        """
        :return: average number of adjectives per sentence
        """
        pos_tags = ["ADJ"]
        total_sent_count = 0
        total_adjs_count = 0
        for sentence in self.doc.sents:
            total_sent_count += 1
            for token in sentence:
                if token.pos_ in pos_tags:
                    total_adjs_count += 1
        if total_sent_count > 0:
            return total_adjs_count/total_sent_count
        else:
            return total_sent_count
    
    def __calculate_adverbs_per_sentence(self) -> float:
        """
        :return: average number of adverbs per sentence
        """
        pos_tags = ["ADV"]
        total_sent_count = 0
        total_advs_count = 0
        for sentence in self.doc.sents:
            total_sent_count += 1
            for token in sentence:
                if token.pos_ in pos_tags:
                    total_advs_count += 1
        if total_sent_count > 0:
            return total_advs_count/total_sent_count
        else:
            return total_sent_count

    def __calculate_vowel_to_consonant_ratio(self) -> float:
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

    def __calculate_capped_to_notcapped_ratio(self, count_remaining=False) -> float:
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

    def __calculate_emoji_count(self, user_verify=False) -> int:
        """
        Count the number of emoji
        :param user_verify: If set to True, additionally return all found emoji for user to verify.
        :return: number of emoji in the text
        """
        emoji_count = emoji.emoji_count(self.raw_text)
        emoji_list = emoji.emoji_list(self.raw_text)
        if user_verify:
            return emoji_count, emoji_list
        return emoji_count
    
    def __calculate_emoticon_count(self, user_verify=False) -> int:
        """
        Count the number of emoticons
        :param user_verify: If set to True, additionally return all found emoticons for user to verify.
        :return: number of emoticons in the text
        """
        emoticon_count = 0
        emoticon_list = []
        for emoticon in EMOTICONS:
            current_count = self.raw_text.count(emoticon)
            if current_count:
                emoticon_list.append(emoticon)
        if user_verify:
            return emoticon_count, emoticon_list
        return emoticon_count
