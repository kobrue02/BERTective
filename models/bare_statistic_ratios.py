import spacy, numpy, emoji

nlp = spacy.load("de_core_news_sm")

# probably near-exhaustive list of vowels possible in any weird spellings of words in a german text
VOWELS = 'aeiouäöüáéíóúàèìòùâêîôûAEIOUÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ'
# probably near-exhaustive list of consonants possible in any weird spellings of words in a german text
CONSONANTS = 'bcdfghjklmnpqrstvwxyzßBCDFGHJKLMNPQRSTVWXYZẞ'

# long list of emoticons
with open(file="./models/wikipedia_emoticons.txt", mode="r", encoding="utf-8") as f:
    EMOTICONS = f.read().split("\n")
# incomplete but probably sufficient-for-training list of chat-like acronyms somewhat likely for german texts
CHAT_ACRONYMS = ["lol", "lul", "lel", "lül", "lmao", "lmfao", "rofl", "smh", "ofc", "nvm", "yolo", "afk", "afaik", "afaic", "tbh", "ngl", "imo", "imho", "idk", "idc", "asap", "bae", "btw", "dafuq", "mmd", "irl", "nsfw", "tldr", "tl,dr", "tl;dr", "tl:dr", "omg", "omfg", "iirc", "asf", "stg"]

# numerals we want to match exactly since with 'in string' they could mess up recall (e.g. Nullnummer, herein, dreist, Achtung …)
NUMERALS_EM = ["null", "ein", "eins", "eines", "eine", "einer", "einem", "einen", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf"]
# numerals we want to match also possibly as part of token since they mostly can be substituted for digits
NUMERALS_IN = ["dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn", "zwanzig", "einundzwanzig", "zweiundzwanzig", "dreiundzwanzig", "vierundzwanzig", "fünfundzwanzig", "sechsundzwanzig", "siebenundzwanzig", "achtundzwanzig", "neunundzwanzig", "dreißig", "einunddreißig", "zweiunddreißig", "dreiunddreißig", "vierunddreißig", "fünfunddreißig", "sechsunddreißig", "siebenunddreißig", "achtunddreißig", "neununddreißig", "vierzig", "einundvierzig", "zweiundvierzig", "dreiundvierzig", "vierundvierzig", "fünfundvierzig", "sechsundvierzig", "siebenundvierzig", "achtundvierzig", "neunundvierzig", "fünfzig", "einundfünfzig", "zweiundfünfzig", "dreiundfünfzig", "vierundfünfzig", "fünfundfünfzig", "sechsundfünfzig", "siebenundfünfzig", "achtundfünfzig", "neunundfünfzig", "sechzig", "einundsechzig", "zweiundsechzig", "dreiundsechzig", "vierundsechzig", "fünfundsechzig", "sechsundsechzig", "siebenundsechzig", "achtundsechzig", "neunundsechzig", "siebzig", "einundsiebzig", "zweiundsiebzig", "dreiundsiebzig", "vierundsiebzig", "fünfundsiebzig", "sechsundsiebzig", "siebenundsiebzig", "achtundsiebzig", "neunundsiebzig", "achtzig", "einundachtzig", "zweiundachtzig", "dreiundachtzig", "vierundachtzig", "fünfundachtzig", "sechsundachtzig", "siebenundachtzig", "achtundachtzig", "neunundachtzig", "neunzig", "einundneunzig", "zweiundneunzig", "dreiundneunzig", "vierundneunzig", "fünfundneunzig", "sechsundneunzig", "siebenundneunzig", "achtundneunzig", "neunundneunzig", "hundert"]

class Statistext:
    def __init__(self, raw_text, mattr_ws=35):
        """
        :param raw_text: string of text for which stats to be calculated
        :param mattr_ws: size (number of tokens) of a window when calculating MATTR,
        consider setting this to the size of the shortest doc in your corpus for every doc,
        although at least something around 30 is recommended for significant TTRs)
        """
        # basic building blocks
        self._raw_text = raw_text
        self._mattr_ws = mattr_ws  # window size for MATTR
        self._doc = nlp(self._raw_text)
        self._sentences = [sent for sent in self._doc.sents]
        self._tokens = [token for token in self._doc]  # (still needed in mattr method)
        #self._types = set([str(token) for token in self._doc])
        self._words = [token for token in self._doc if not token.is_digit and not token.is_space and not token.is_punct]
        self._numbers = [token for token in self._doc if token.is_digit]
        self._numerals = [token for token in self._doc if str(token) in NUMERALS_EM or any(num for num in NUMERALS_IN if num in str(token))]
        self._nouns = [token for token in self._doc if token.pos_ == "NOUN"]
        self._verbs = [token for token in self._doc if token.pos_ == "VERB"]
        self._adjectives = [token for token in self._doc if token.pos_ == "ADJ"]
        self._adverbs = [token for token in self._doc if token.pos_ == "ADV"]
        self._determiners = [token for token in self._doc if token.pos_ == "DET"]
        self._vowels = [char for char in self._raw_text if char in VOWELS]
        self._consonants = [char for char in self._raw_text if char in CONSONANTS]
        self._nocap = [word for word in self._words if str(word)[0].islower()]
        self._cap = [word for word in self._words if str(word)[0].isupper()]
        self._emoticons = [token for token in self._raw_text.split() if any(e for e in EMOTICONS if e == token)]
        self._chat_acronyms = [word for word in self._words if str(word).lower() in CHAT_ACRONYMS]
        # lexical diversity
        self.mattr = self._calculate_mattr()
        # average length of 'lexical' words
        self.characters_per_word = sum(len(word) for word in self._words) / len(self._words) if len(self._words) > 0 else 0.0
        self.characters_per_noun = sum(len(noun) for noun in self._nouns) / len(self._nouns) if len(self._nouns) > 0 else 0.0
        self.characters_per_verb = sum(len(verb) for verb in self._verbs) / len(self._verbs) if len(self._verbs) > 0 else 0.0
        self.characters_per_adjective = sum(len(adj) for adj in self._adjectives) / len(self._adjectives) if len(self._adjectives) > 0 else 0.0
        self.characters_per_adverb = sum(len(adv) for adv in self._adverbs) / len(self._adverbs) if len(self._adverbs) > 0 else 0.0
        # average sentence content of 'lexical' words
        self.words_per_sentence = len(self._words) / len(self._sentences) if len(self._sentences) > 0 else 0.0
        self.nouns_per_sentence = len(self._nouns) / len(self._sentences) if len(self._sentences) > 0 else 0.0
        self.verbs_per_sentence = len(self._verbs) / len(self._sentences) if len(self._sentences) > 0 else 0.0
        self.adjectives_per_sentence = len(self._adjectives) / len(self._sentences) if len(self._sentences) > 0 else 0.0
        self.adverbs_per_sentence = len(self._adverbs) / len(self._sentences) if len(self._sentences) > 0 else 0.0
        # pos/pos ratios
        self.articles_per_noun = len(self._determiners) / len(self._nouns) if len(self._nouns) > 0 else 0.0
        self.adjectives_per_noun = len(self._adjectives) / len(self._nouns) if len(self._nouns) > 0 else 0.0
        self.adverbs_per_verb = len(self._adverbs) / len(self._verbs) if len(self._verbs) > 0 else 0.0
        # special register symbol counts
        self.emoji_count = emoji.emoji_count(self._raw_text)
        self.emoticon_count = len(self._emoticons)
        self.chat_acronym_count = len(self._chat_acronyms)
        # dings
        self.vowel_to_consonant_ratio = len(self._vowels) / len(self._consonants) if len(self._consonants) > 0 else 0.0
        self.capped_to_notcapped_ratio = len(self._cap) / len(self._nocap) if len(self._nocap) > 0 else 0.0
        self.number_representation = len(self._numbers) / len(self._numerals) if len(self._numerals) > 0 else 0.0
        # bums
        self.punctuation_variation = "TODO: def"  # braucht liste
        self.units_representation = "TODO: def"  # braucht liste
        self.anglicism_style = "TODO: def"  # braucht liste
        self.num_unit_spacing = "TODO: def"  # braucht liste(n) und re – s. u.
        self.fillers_share = "TODO: def"  # braucht liste
        self.colloq_alt_spelling = "TODO: def"  # braucht liste
        # all working stats
        self.all_stats = numpy.array([
            self.mattr,
            self.characters_per_word, self.characters_per_noun, self.characters_per_verb, self.characters_per_adjective, self.characters_per_adverb,
            self.words_per_sentence, self.nouns_per_sentence, self.verbs_per_sentence, self.adjectives_per_sentence, self.adverbs_per_sentence,
            self.articles_per_noun, self.adjectives_per_noun, self.adverbs_per_verb,
            self.emoji_count, self.emoticon_count, self.chat_acronym_count,
            self.vowel_to_consonant_ratio, self.capped_to_notcapped_ratio, self.number_representation
            ])
    
    def _calculate_mattr(self) -> float:
        doc_size = len(self._tokens)
        window_size = self._mattr_ws
        window_start = 0
        ttr_values = []
        for i in range(doc_size//window_size):
            window = slice(window_start, window_start+window_size)
            window_doc = self._doc[window]
            window_tokens = [str(token) for token in window_doc]
            window_types = set(window_tokens)
            window_ttr = len(window_types)/len(window_tokens) if len(window_tokens) > 0 else 0.0
            ttr_values.append(window_ttr)
            window_start += window_size
        return sum(ttr_values)/len(ttr_values) if len(ttr_values) > 0 else 0.0
    