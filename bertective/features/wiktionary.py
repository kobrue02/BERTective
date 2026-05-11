"""Wiktionary-based lexical feature extraction."""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import TYPE_CHECKING

import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm

from bertective.constants import WIKTIONARY_JSON, WIKTIONARY_PARQUET, WIKTIONARY_PICKLE
from bertective.exceptions import ResourceNotFoundError

if TYPE_CHECKING:
    from bertective.corpus import DataCorpus

logger = logging.getLogger(__name__)


class WiktionaryModel:
    """Generate 27-dimensional Wiktionary feature vectors.

    Each dimension counts how many tokens in a text appear in one of the 27
    Wiktionary word lists (Anglicisms, abbreviations, technical terms, dialects, …).

    Usage::

        # Build from a corpus (slow, generates wiktionary.parquet)
        model = WiktionaryModel(source=corpus)
        model.df_matrix.to_parquet("data/wiktionary/wiktionary.parquet")

        # Load pre-built matrix
        model = WiktionaryModel(path="data/wiktionary/wiktionary.parquet")
        vector = model[doc_id]          # numpy array of shape (27,)

        # One-off embedding for inference
        model = WiktionaryModel()
        _, vector = model.get_matches("some german text")
    """

    def __init__(
        self,
        path: str | Path | None = None,
        source: "DataCorpus | None" = None,
    ) -> None:
        if not WIKTIONARY_JSON.exists():
            raise ResourceNotFoundError(
                f"Wiktionary word list not found at {WIKTIONARY_JSON}. "
                "Run `bertective data download --wiktionary` first."
            )
        with open(WIKTIONARY_JSON, "r", encoding="utf-8") as fh:
            self.wiktionary: dict[str, list[str]] = json.load(fh)

        self.vectors: dict[int, np.ndarray] = {}

        if path:
            self.df_matrix = pd.read_parquet(path)
            self.vectors = self._vectors_from_df()

        if source:
            self._matrix = self._build_matrix(source)
            self.df_matrix = self._to_df()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_matches(self, text: str) -> tuple[dict[str, int], np.ndarray]:
        """Return a per-category match count dict and its numpy vector form."""
        tokens = set(nltk.word_tokenize(text.lower()))
        dist = {k: sum(1 for w in words if w.lower() in tokens) for k, words in self.wiktionary.items()}
        return dist, self._to_vector(dist)

    def __getitem__(self, doc_id: int) -> np.ndarray:
        return self.vectors[doc_id]

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_matrix(self, source: "DataCorpus") -> dict:
        logger.info("Building Wiktionary matrix…")
        matrix: dict = {}
        for obj in tqdm(source):
            dist, vector = self.get_matches(obj.text)
            matrix[obj.id] = dist
            self.vectors[obj.id] = vector
        return matrix

    def _to_df(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["DataObject_ID"] = list(self._matrix.keys())
        for key in self.wiktionary:
            df[key] = [row[key] for row in self._matrix.values()]
        return df

    def _to_vector(self, dist: dict[str, int]) -> np.ndarray:
        return np.array(list(dist.values()))

    def _vectors_from_df(self) -> dict[int, np.ndarray]:
        if WIKTIONARY_PICKLE.exists():
            logger.info("Loading cached Wiktionary vectors…")
            with open(WIKTIONARY_PICKLE, "rb") as fh:
                return pickle.load(fh)

        logger.info("Building Wiktionary vectors from parquet…")
        columns = self.df_matrix.columns.tolist()[1:]
        vectors = {
            i: np.array([self.df_matrix[col].iloc[i] for col in columns])
            for i in tqdm(range(len(self.df_matrix)))
        }
        with open(WIKTIONARY_PICKLE, "wb") as fh:
            pickle.dump(vectors, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return vectors
