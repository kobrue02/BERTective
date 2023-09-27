import spacy, numpy, emoji

nlp = spacy.load("de_core_news_sm")
VOWELS = 'aeiouäöüáéíóúàèìòùâêîôûAEIOUÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ'
CONSONANTS = 'bcdfghjklmnpqrstvwxyzßBCDFGHJKLMNPQRSTVWXYZẞ'
with open(file="wikipedia_emoticons.txt", mode="r", encoding='utf-8') as f:
    EMOTICONS = f.read().split("\n")  # FIXME: make list not contain funny unicode escape seqs anymore

CHAT_ACRONYMS = ["lol", "lmao", "gtg", "gg", "wp", "gl", "hf", "glhf", "gl;hf", "lmfao"]  # TODO: fill uppp (few suffice bc only 4 training)

class Statistext:
    def __init__(self, raw_text):
        # basic building blocks
        self._raw_text = raw_text
        self._doc = nlp(self._raw_text)
        self._sentences = [sent for sent in self._doc.sents]
        self._tokens = [token for token in self._doc]
        self._types = set([str(token) for token in self._doc])  # currently only used for ttr
        self._words = [token for token in self._doc if not token.is_digit and not token.is_space and not token.is_punct]
        self._numerals = [token for token in self._doc if token.is_digit] #or token in NUMERALS]  # FIXME: spaCy bug? Token.like_num does not include real numerals (words for numbers), neither German nor English, even though it should according to official documentation
        self._nouns = [token for token in self._doc if token.pos_ == "NOUN"]
        self._verbs = [token for token in self._doc if token.pos_ == "VERB"]
        self._adjectives = [token for token in self._doc if token.pos_ == "ADJ"]
        self._adverbs = [token for token in self._doc if token.pos_ == "ADV"]
        self._determiners = [token for token in self._doc if token.pos_ == "DET"]
        self._vowels = [char for char in self._raw_text if char in VOWELS]
        self._consonants = [char for char in self._raw_text if char in CONSONANTS]
        self._nocap = [word for word in self._words if str(word)[0].islower()]
        self._cap = [word for word in self._words if str(word)[0].isupper()]
        self._chat_acronyms = [word for word in self._words if str(word) in CHAT_ACRONYMS]
        # lexical diversity
        try:
            self.ttr = len(self._types)/len(self._tokens)
        except ZeroDivisionError:
            self.ttr = 0
        try:
            self.mattr = "TODO: def"
        except ZeroDivisionError:
            self.mattr = 0  # TODO: check if even necessary here (depending on method content)
        # average length of 'lexical' words
        try:
            self.characters_per_word = sum(len(word) for word in self._words) / len(self._words)
        except ZeroDivisionError:
            self.characters_per_word = 0
        try:
            self.characters_per_noun = sum(len(noun) for noun in self._nouns) / len(self._nouns)
        except ZeroDivisionError:
            self.characters_per_noun = 0
        try:
            self.characters_per_verb = sum(len(verb) for verb in self._verbs) / len(self._verbs)
        except ZeroDivisionError:
            self.characters_per_verb = 0
        try:
            self.characters_per_adjective = sum(len(adj) for adj in self._adjectives) / len(self._adjectives)
        except ZeroDivisionError:
            self.characters_per_adjective = 0
        try:
            self.characters_per_adverb = sum(len(adv) for adv in self._adverbs) / len(self._adverbs)
        except ZeroDivisionError:
            self.characters_per_adverb = 0
        # average sentence content of 'lexical' words
        try:
            self.words_per_sentence = len(self._words) / len(self._sentences)
            self.nouns_per_sentence = len(self._nouns) / len(self._sentences)
            self.verbs_per_sentence = len(self._verbs) / len(self._sentences)
            self.adjectives_per_sentence = len(self._adjectives) / len(self._sentences)
            self.adverbs_per_sentence = len(self._adverbs) / len(self._sentences)
        except ZeroDivisionError:
            self.words_per_sentence = 0
            self.nouns_per_sentence = 0
            self.verbs_per_sentence = 0
            self.adjectives_per_sentence = 0
            self.adverbs_per_sentence = 0
        # pos/pos ratios
        try:
            self.articles_per_noun = len(self._determiners) / len(self._nouns)
            self.adjectives_per_noun = len(self._adjectives) / len(self._nouns)
        except ZeroDivisionError:
            self.articles_per_noun = 0
            self.adjectives_per_noun = 0
        try:
            self.adverbs_per_verb = len(self._adverbs) / len(self._verbs)
        except ZeroDivisionError:
            self.adverbs_per_verb = 0
        # special register symbol counts
        self.emoji_count = emoji.emoji_count(self._raw_text)
        self.emoticon_count = sum(self._raw_text.count(emoticon) for emoticon in EMOTICONS)
        self.chat_acronym_count = len(self._chat_acronyms)
        # dings
        try:
            self.vowel_to_consonant_ratio = len(self._vowels) / len(self._consonants)
        except ZeroDivisionError:
            self.vowel_to_consonant_ratio = 0
        try:
            self.capped_to_notcapped_ratio = len(self._cap) / len(self._nocap)
        except ZeroDivisionError:
            self.capped_to_notcapped_ratio = 0
        # bums
        self.punctuation_variation = "TODO: def"  # braucht liste
        self.number_representation = "TODO: def"  # braucht liste
        self.units_representation = "TODO: def"  # braucht liste
        self.anglicism_style = "TODO: def"  # braucht liste
        self.num_unit_spacing = "TODO: def"  # braucht liste(n) und re – s. u.
        self.fillers_share = "TODO: def"  # braucht liste
        self.colloq_alt_spelling = "TODO: def"  # braucht liste
        # all working stats
        self.all_stats = numpy.array([
            #self.ttr, self.mattr,
            self.characters_per_word, self.characters_per_noun, self.characters_per_verb, self.characters_per_adjective, self.characters_per_adverb,
            self.words_per_sentence, self.nouns_per_sentence, self.verbs_per_sentence, self.adjectives_per_sentence, self.adverbs_per_sentence,
            self.articles_per_noun, self.adjectives_per_noun, self.adverbs_per_verb,
            self.emoji_count, self.emoticon_count, self.chat_acronym_count,
            self.vowel_to_consonant_ratio, self.capped_to_notcapped_ratio#,
            #self.punctuation_variation, self.number_representation, self.units_representation, self.anglicism_style, self.num_unit_spacing, self.fillers_share, self.colloq_alt_spelling
            ])
