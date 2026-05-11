"""Statistical linguistic features extracted via spaCy."""
from __future__ import annotations

import numpy as np
import emoji
import spacy

from bertective.constants import EMOTICONS_FILE

# ---------------------------------------------------------------------------
# Module-level constants (lazy-loaded where possible)
# ---------------------------------------------------------------------------

VOWELS = "aeiouäöüáéíóúàèìòùâêîôûAEIOUÄÖÜÁÉÍÓÚÀÈÌÒÙÂÊÎÔÛ"
CONSONANTS = "bcdfghjklmnpqrstvwxyzßBCDFGHJKLMNPQRSTVWXYZẞ"

CHAT_ACRONYMS: list[str] = [
    "lol", "lul", "lel", "lül", "lmao", "lmfao", "rofl", "smh", "ofc", "nvm",
    "yolo", "afk", "afaik", "afaic", "tbh", "ngl", "imo", "imho", "idk", "idc",
    "asap", "bae", "btw", "dafuq", "mmd", "irl", "nsfw", "tldr", "tl,dr",
    "tl;dr", "tl:dr", "omg", "omfg", "iirc", "asf", "stg",
]

NUMERALS_EXACT: list[str] = [
    "null", "ein", "eins", "eines", "eine", "einer", "einem", "einen",
    "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun",
    "zehn", "elf", "zwölf",
]
NUMERALS_SUBSTR: list[str] = [
    "dreizehn", "vierzehn", "fünfzehn", "sechzehn", "siebzehn", "achtzehn",
    "neunzehn", "zwanzig", "einundzwanzig", "zweiundzwanzig", "dreiundzwanzig",
    "vierundzwanzig", "fünfundzwanzig", "sechsundzwanzig", "siebenundzwanzig",
    "achtundzwanzig", "neunundzwanzig", "dreißig", "vierzig", "fünfzig",
    "sechzig", "siebzig", "achtzig", "neunzig", "hundert",
]

_nlp: spacy.Language | None = None
_emoticons: list[str] | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_sm")
    return _nlp


def _get_emoticons() -> list[str]:
    global _emoticons
    if _emoticons is None:
        _emoticons = EMOTICONS_FILE.read_text(encoding="utf-8").split("\n")
    return _emoticons


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Statistext:
    """Compute a fixed-dimensional linguistic statistics vector for a German text.

    The ``all_stats`` attribute is a 20-element numpy array ready for model input.
    """

    def __init__(self, raw_text: str, mattr_ws: int = 35) -> None:
        """
        :param raw_text: German text to analyse.
        :param mattr_ws: Window size for MATTR (Moving-Average Type–Token Ratio).
            Set to the size of the shortest document in your corpus; ≥30 is recommended.
        """
        nlp = _get_nlp()
        emoticons = _get_emoticons()

        self._raw_text = raw_text
        self._mattr_ws = mattr_ws
        self._doc = nlp(raw_text)

        self._sentences = list(self._doc.sents)
        self._tokens = list(self._doc)
        self._words = [t for t in self._doc if not t.is_digit and not t.is_space and not t.is_punct]
        self._numbers = [t for t in self._doc if t.is_digit]
        self._numerals = [
            t for t in self._doc
            if str(t) in NUMERALS_EXACT or any(n in str(t) for n in NUMERALS_SUBSTR)
        ]
        self._nouns = [t for t in self._doc if t.pos_ == "NOUN"]
        self._verbs = [t for t in self._doc if t.pos_ == "VERB"]
        self._adjectives = [t for t in self._doc if t.pos_ == "ADJ"]
        self._adverbs = [t for t in self._doc if t.pos_ == "ADV"]
        self._determiners = [t for t in self._doc if t.pos_ == "DET"]
        self._vowels = [c for c in raw_text if c in VOWELS]
        self._consonants = [c for c in raw_text if c in CONSONANTS]
        self._nocap = [w for w in self._words if str(w)[0].islower()]
        self._cap = [w for w in self._words if str(w)[0].isupper()]
        self._emoticons = [t for t in raw_text.split() if t in emoticons]
        self._chat_acronyms = [w for w in self._words if str(w).lower() in CHAT_ACRONYMS]

        n_words = len(self._words) or 1
        n_sents = len(self._sentences) or 1
        n_nouns = len(self._nouns) or 1
        n_verbs = len(self._verbs) or 1
        n_cons = len(self._consonants) or 1
        n_nocap = len(self._nocap) or 1
        n_numerals = len(self._numerals) or 1

        # Lexical diversity
        self.mattr = self._calculate_mattr()

        # Average character length per POS class
        self.characters_per_word = sum(len(w) for w in self._words) / n_words
        self.characters_per_noun = sum(len(n) for n in self._nouns) / n_nouns
        self.characters_per_verb = sum(len(v) for v in self._verbs) / (len(self._verbs) or 1)
        self.characters_per_adjective = sum(len(a) for a in self._adjectives) / (len(self._adjectives) or 1)
        self.characters_per_adverb = sum(len(a) for a in self._adverbs) / (len(self._adverbs) or 1)

        # POS density per sentence
        self.words_per_sentence = len(self._words) / n_sents
        self.nouns_per_sentence = len(self._nouns) / n_sents
        self.verbs_per_sentence = len(self._verbs) / n_sents
        self.adjectives_per_sentence = len(self._adjectives) / n_sents
        self.adverbs_per_sentence = len(self._adverbs) / n_sents

        # POS / POS ratios
        self.articles_per_noun = len(self._determiners) / n_nouns
        self.adjectives_per_noun = len(self._adjectives) / n_nouns
        self.adverbs_per_verb = len(self._adverbs) / (len(self._verbs) or 1)

        # Register markers
        self.emoji_count = emoji.emoji_count(raw_text)
        self.emoticon_count = len(self._emoticons)
        self.chat_acronym_count = len(self._chat_acronyms)

        # Phonological / orthographic ratios
        self.vowel_to_consonant_ratio = len(self._vowels) / n_cons
        self.capped_to_notcapped_ratio = len(self._cap) / n_nocap
        self.number_representation = len(self._numbers) / n_numerals

        self.all_stats = np.array([
            self.mattr,
            self.characters_per_word, self.characters_per_noun,
            self.characters_per_verb, self.characters_per_adjective,
            self.characters_per_adverb,
            self.words_per_sentence, self.nouns_per_sentence,
            self.verbs_per_sentence, self.adjectives_per_sentence,
            self.adverbs_per_sentence,
            self.articles_per_noun, self.adjectives_per_noun,
            self.adverbs_per_verb,
            self.emoji_count, self.emoticon_count, self.chat_acronym_count,
            self.vowel_to_consonant_ratio, self.capped_to_notcapped_ratio,
            self.number_representation,
        ])

    def _calculate_mattr(self) -> float:
        """Moving-Average Type–Token Ratio over non-overlapping windows."""
        doc_size = len(self._tokens)
        ws = self._mattr_ws
        ttr_values = [
            len(set(str(t) for t in self._doc[i * ws:(i + 1) * ws])) / ws
            for i in range(doc_size // ws)
        ]
        return sum(ttr_values) / len(ttr_values) if ttr_values else 0.0
