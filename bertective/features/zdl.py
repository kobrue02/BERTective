"""ZDL Regionalkorpus feature extraction.

ZDLVectorModel  — legacy sklearn-based classifier (mostly superseded).
ZDLVectorMatrix — batch vectoriser used during corpus preprocessing.
"""
from __future__ import annotations

import json
import logging
import re
import string
import time
from pathlib import Path
from threading import Thread
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import requests
import spacy
from tqdm import tqdm

from bertective.constants import AREAL_DICT, ZDL_VECTOR_DICT, STOPWORDS_CACHE

if TYPE_CHECKING:
    from bertective.corpus import DataCorpus, DataObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded resources
# ---------------------------------------------------------------------------

_nlp: spacy.Language | None = None
_stopwords: list[str] | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_md")
    return _nlp


def _get_stopwords() -> list[str]:
    """Return German stopwords, downloading and caching them if needed."""
    global _stopwords
    if _stopwords is not None:
        return _stopwords

    if STOPWORDS_CACHE.exists():
        _stopwords = STOPWORDS_CACHE.read_text(encoding="utf-8").splitlines()
        return _stopwords

    words: list[str] = []
    try:
        import pandas as _pd
        df = _pd.read_csv(
            "https://zenodo.org/record/3995594/files/SW-DE-RS_v1-0-0_Datensatz.csv?download=1"
        )
        for col in df.columns:
            words += [w for w in df[col].tolist() if not isinstance(w, float)]
    except Exception:
        logger.warning("Could not download extended stopword list; falling back to NLTK.")

    try:
        from nltk.corpus import stopwords as nltk_sw
        words += nltk_sw.words("german")
    except LookupError:
        pass

    _stopwords = list(set(words))
    STOPWORDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    STOPWORDS_CACHE.write_text("\n".join(_stopwords), encoding="utf-8")
    return _stopwords


# ---------------------------------------------------------------------------
# Tokenisation helper
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    """Deduplicated, stopword-filtered tokens from *text*."""
    import nltk
    stopwords = _get_stopwords()
    unique = list(dict.fromkeys(nltk.word_tokenize(text)))
    return [
        t for t in unique
        if t.lower() not in stopwords
        and t not in string.punctuation
        and len(t) <= 25
    ]


# ---------------------------------------------------------------------------
# ZDL API
# ---------------------------------------------------------------------------

def zdl_request(query: str, corpus: str = "regional", by: str = "areal", fmt: str = "json") -> list[dict]:
    url = f"https://www.dwds.de/api/ppm?q={query}&corpus={corpus}&by={by}&format={fmt}"
    return requests.get(url).json()


# ---------------------------------------------------------------------------
# ZDLVectorModel  (legacy sklearn-based, kept for compatibility)
# ---------------------------------------------------------------------------

class ZDLVectorModel:
    """Sklearn-based regiolect classifier using ZDL regionalkorpus token embeddings.

    Mostly superseded by the deep-learning pipeline; the static ``_vectorize_sample``
    method is still used by ``ZDLVectorMatrix`` and the predictor.
    """

    def __init__(
        self,
        read_pickle: bool,
        classifier,
        locale_type: str = "all",
        path: str = "data",
    ) -> None:
        if locale_type not in ("all", "EAST_WEST", "NORTH_SOUTH"):
            raise ValueError('locale_type must be "all", "EAST_WEST", or "NORTH_SOUTH"')
        if not isinstance(read_pickle, bool):
            raise TypeError("read_pickle must be True or False")

        self.path = path
        self.locale_type = locale_type
        self.areal_dict = AREAL_DICT
        self.classifier = classifier

        with open(ZDL_VECTOR_DICT, "r", encoding="utf-8") as fh:
            self.vector_dict: dict = json.load(fh)

        data = pd.read_pickle("vectors/zdl_vector_matrix.pickle") if read_pickle else self._train()
        self.training_data = self._create_training_set(data)
        self.model = self._fit()

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def _create_training_set(self, data: pd.DataFrame) -> pd.DataFrame:
        max_len = max(len(v) for v in data["vector"])
        data["padded_vectors"] = [self._zero_pad(v, max_len) for v in data["vector"]]
        data["simple_locale"] = data["LOCALE"].map(self._simplify_locale)
        data["LOCALE_NUM"] = data["LOCALE"].map(self._locale_to_num)
        return data

    def _fit(self):
        from sklearn.model_selection import train_test_split
        X = self.training_data["padded_vectors"].tolist()
        y = self.training_data["LOCALE_NUM" if self.locale_type == "all" else "simple_locale"]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.1, random_state=42, stratify=y
        )
        self.classifier.fit(self.X_train, self.y_train)
        return self.classifier

    def _train(self) -> pd.DataFrame:
        import os
        rows = []
        locale_dir = Path(self.path) / "reddit" / "locales"
        for fname in tqdm(list(locale_dir.iterdir())):
            if fname.suffix != ".json":
                continue
            with open(fname, "r", encoding="utf-8") as fh:
                file_data = json.load(fh)
            city = fname.stem
            for areal, cities in self.areal_dict.items():
                if city in cities:
                    texts = list({
                        item["selftext"] for item in file_data["data"]
                        if self._is_german(item["selftext"])
                    })
                    rows.append(pd.DataFrame({"texts": texts, "LOCALE": areal}))
        df = pd.concat(rows, ignore_index=True)
        df["vector"] = [
            self._vectorize_sample(t, self.vector_dict, verbose=False)[0]
            for t in tqdm(df["texts"])
        ]
        df.to_pickle("vectors/zdl_vector_matrix.pickle")
        with open(ZDL_VECTOR_DICT, "w", encoding="utf-8") as fh:
            json.dump(self.vector_dict, fh)
        return df

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> None:
        from sklearn.metrics import classification_report
        import seaborn as sns
        import matplotlib.pyplot as plt

        y_pred = self.model.predict(self.X_test)
        report = classification_report(
            self.y_test,
            y_pred,
            output_dict=True,
            target_names=list(AREAL_DICT.keys()) if self.locale_type == "all" else None,
        )
        sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Greens")
        plt.show()

    # ------------------------------------------------------------------
    # Static vectoriser — used by ZDLVectorMatrix and the predictor
    # ------------------------------------------------------------------

    @staticmethod
    def _json_to_vector(response: list[dict]) -> np.ndarray:
        return np.array([item["ppm"] for item in response])

    @staticmethod
    def _vectorize_sample(
        text: str,
        vectionary: dict,
        verbose: bool = False,
    ) -> tuple[np.ndarray, dict]:
        """Tokenise *text* and embed each token as a 6-d ZDL regional vector.

        Returns ``(matrix, updated_vectionary)``.
        """
        text = re.sub(r"<.*?>", "", text.replace("\n", ""))
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        nlp = _get_nlp()
        vectors: list[np.ndarray] = []

        for token in tokenize(text):
            if "http" in token:
                continue
            if token in vectionary:
                v = vectionary[token]
            elif token.lower() in vectionary:
                v = vectionary[token.lower()]
            else:
                lemma = nlp(token)[0].lemma_
                if lemma in vectionary:
                    v = vectionary[lemma]
                else:
                    try:
                        response = zdl_request(lemma)
                        if verbose:
                            tqdm.write(f'ZDL API call for "{lemma}"')
                    except (requests.exceptions.JSONDecodeError, ValueError):
                        continue
                    v = ZDLVectorModel._json_to_vector(response)
                    vectionary[lemma] = v.tolist()
            vectors.append(np.asarray(v))

        complete = [v for v in vectors if len(v) == 6]
        return np.array(complete) if complete else np.zeros((0, 6)), vectionary

    # ------------------------------------------------------------------
    # Misc helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_german(text: str) -> bool:
        from langdetect import detect, DetectorFactory, lang_detect_exception
        DetectorFactory.seed = 0
        if text in ("[removed]", "[deleted]") or len(text) < 20 or len(text.split()) < 3:
            return False
        try:
            return detect(text) == "de"
        except lang_detect_exception.LangDetectException:
            return False

    def _zero_pad(self, vector: np.ndarray, max_len: int) -> np.ndarray:
        diff = max_len - len(vector)
        return np.append(vector, [np.zeros(6)] * diff)

    def _simplify_locale(self, locale: str) -> str:
        mapping = {
            "NORTH_SOUTH": {
                "DE-NORTH": ["DE-NORTH-WEST", "DE-NORTH-EAST"],
                "DE-SOUTH": ["DE-SOUTH-WEST", "DE-SOUTH-EAST", "DE-MIDDLE-WEST", "DE-MIDDLE-EAST"],
            },
            "EAST_WEST": {
                "DE-EAST": ["DE-NORTH-EAST", "DE-MIDDLE-EAST", "DE-SOUTH-EAST"],
                "DE-WEST": ["DE-NORTH-WEST", "DE-MIDDLE-WEST", "DE-SOUTH-WEST"],
            },
        }
        if self.locale_type == "all":
            return locale
        for key, regions in mapping[self.locale_type].items():
            if locale in regions:
                return key
        return locale

    @staticmethod
    def _locale_to_num(locale: str) -> int:
        return {
            "DE-NORTH-EAST": 1, "DE-NORTH-WEST": 2,
            "DE-MIDDLE-EAST": 3, "DE-MIDDLE-WEST": 4,
            "DE-SOUTH-EAST": 5, "DE-SOUTH-WEST": 6,
        }[locale]


# ---------------------------------------------------------------------------
# ZDLVectorMatrix — batch vectoriser for DataCorpus
# ---------------------------------------------------------------------------

class ZDLVectorMatrix:
    """Generate a ZDL vector per document in a DataCorpus, using threading for speed."""

    def __init__(
        self,
        source: "DataCorpus | None" = None,
        path: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.data = source
        self.verbose = verbose

        try:
            with open(ZDL_VECTOR_DICT, "r", encoding="utf-8") as fh:
                self.vectionary: dict = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            self.vectionary = {}

        if path:
            with open("data/vectors/ZDLCorpus_data.json", "r") as fh:
                self.vectors: dict = json.load(fh)
        else:
            self.vectors = self._vectorize_all()

    # ------------------------------------------------------------------
    # Vectorisation
    # ------------------------------------------------------------------

    def _vectorize_all(self) -> dict:
        if len(self.data) < 128:
            logger.info("Small corpus: using single thread.")
            return self._vectorize_sequential()
        return self._vectorize_threaded()

    def _vectorize_sequential(self) -> dict:
        matrix: dict = {}
        for obj in tqdm(self.data):
            v, self.vectionary = ZDLVectorModel._vectorize_sample(
                obj.text, self.vectionary, verbose=self.verbose
            )
            matrix[obj.id] = v
        return matrix

    def _vectorize_threaded(self) -> dict:
        logger.info("Building ZDL matrix with %d threads…", cpu_count())
        self._chunk_matrices: list[dict] = []
        self._temp_vectionaries: list[dict] = []

        n = len(self.data)
        cores = cpu_count()
        chunk = n // cores
        slices = [slice(i * chunk, (i + 1) * chunk if i < cores - 1 else None) for i in range(cores)]

        threads = [Thread(target=self._batch, args=(s,)) for s in slices]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for tmp in self._temp_vectionaries:
            self.vectionary.update(tmp)
        ZDL_VECTOR_DICT.parent.mkdir(parents=True, exist_ok=True)
        with open(ZDL_VECTOR_DICT, "w", encoding="utf-8") as fh:
            json.dump(self.vectionary, fh)

        matrix: dict = {}
        for chunk_dict in self._chunk_matrices:
            matrix.update(chunk_dict)
        return matrix

    def _batch(self, sl: slice) -> None:
        batch: dict = {}
        for obj in tqdm(self.data[sl]):
            v, vectionary = ZDLVectorModel._vectorize_sample(
                obj.text, self.vectionary, verbose=self.verbose
            )
            batch[obj.id] = v
            self._temp_vectionaries.append(vectionary)
        self._chunk_matrices.append(batch)

    def save_to_json(self, path: str = "data/vectors/ZDLCorpus_data.json") -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.vectors, fh)
