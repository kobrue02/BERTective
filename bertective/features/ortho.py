"""Orthography and spelling-error feature extraction.

Matches text against orthography and error lists sourced from korrekturen.de
and embeds matches via spaCy's tok2vec vectors.  Generates 5 embeddings per
text, each of shape (96,), representing different orthography variants
(ancient / revolutionized / modern / error / correct).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import spacy

from bertective.constants import ORTHOGRAPHY_JSON, ERROR_TUPLES_JSON
from bertective.exceptions import ResourceNotFoundError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_nlp: spacy.Language | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("de_core_news_sm")
    return _nlp


class OrthoMatrixModel:
    """Generate orthography / spelling-error feature vectors for German texts.

    Requires ``data/annotation/orthography.json`` and
    ``data/annotation/error_tuples.json`` to exist.
    Run ``bertective data download`` to generate them.
    """

    def __init__(self) -> None:
        self.orthography: dict = self._load_json(ORTHOGRAPHY_JSON, "orthography")
        self.error_tuples: dict = self._load_json(ERROR_TUPLES_JSON, "error_tuples")

    # ------------------------------------------------------------------
    # Public embedding API
    # ------------------------------------------------------------------

    def find_ortho_match_in_text(self, text: str, orthography_set: str) -> np.ndarray:
        """Average spaCy tok2vec embeddings for tokens that match *orthography_set*.

        :param text: German text to embed.
        :param orthography_set: Key into the orthography JSON
            (``"ancient"``, ``"revolutionized"``, or ``"modern"``).
        :returns: Averaged embedding of shape ``(96,)``, or zeros if no match.
        """
        matched = [
            _get_nlp()(item).vector
            for item in self.orthography["orthographies"].get(orthography_set, [])
            if item and item in text
        ]
        return self._average(matched)

    def find_error_match_in_text(self, text: str, reference_set: str) -> np.ndarray:
        """Average spaCy tok2vec embeddings for tokens that match a spelling list.

        :param text: German text to embed.
        :param reference_set: ``"error"`` (misspellings) or ``"correct"`` (correct forms).
        :returns: Averaged embedding of shape ``(96,)``, or zeros if no match.
        """
        if reference_set not in ("error", "correct"):
            raise ValueError("reference_set must be 'error' or 'correct'")
        idx = 0 if reference_set == "error" else 1
        matched = [
            _get_nlp()(entry[idx]).vector
            for entry in self.error_tuples["errors"]
            if entry[idx] and entry[idx] in text
        ]
        return self._average(matched)

    def embed(self, text: str) -> dict[str, list[float]]:
        """Compute all five orthography embeddings for *text*.

        Returns a dict with keys
        ``embedding_ancient``, ``embedding_revolutionized``, ``embedding_modern``,
        ``embedding_error``, ``embedding_correct``.
        """
        return {
            "embedding_ancient":       self.find_ortho_match_in_text(text, "ancient").tolist(),
            "embedding_revolutionized": self.find_ortho_match_in_text(text, "revolutionized").tolist(),
            "embedding_modern":        self.find_ortho_match_in_text(text, "modern").tolist(),
            "embedding_error":         self.find_error_match_in_text(text, "error").tolist(),
            "embedding_correct":       self.find_error_match_in_text(text, "correct").tolist(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _average(vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            return np.zeros(96)
        result = np.zeros(96)
        for v in vectors:
            result = np.add(result, v)
        return result

    @staticmethod
    def _load_json(path: Path, name: str) -> dict:
        if not path.exists():
            raise ResourceNotFoundError(
                f"Required annotation file not found: {path}. "
                "Run `bertective data download` to generate it."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
