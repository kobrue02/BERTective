"""Author-profile inference from pre-trained models."""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.models import load_model

from bertective.constants import TRAINED_MODELS_DIR, ZDL_VECTOR_DICT
from bertective.exceptions import ParsedPathNotExistError
from bertective.features.ortho import OrthoMatrixModel
from bertective.features.stats import Statistext
from bertective.features.wiktionary import WiktionaryModel
from bertective.features.zdl import ZDLVectorModel
from bertective.models.trainer import floats_to_labels, zero_pad_zdl

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction helpers (single-sample)
# ---------------------------------------------------------------------------

def _zdl_embedding(text: str) -> np.ndarray:
    with open(ZDL_VECTOR_DICT, "r", encoding="utf-8") as fh:
        vectionary = json.load(fh)
    vector, _ = ZDLVectorModel._vectorize_sample(text, vectionary, verbose=False)
    return vector


def _ortho_embeddings(text: str) -> list:
    ortho = OrthoMatrixModel()
    return list(ortho.embed(text).values())


def _wiktionary_embedding(text: str) -> np.ndarray:
    wm = WiktionaryModel()
    _, vector = wm.get_matches(text)
    return vector


def _stats_embedding(text: str) -> np.ndarray:
    return Statistext(text).all_stats


def _concat_features(
    zdl: np.ndarray,
    ortho: list,
    wikt: np.ndarray,
    stat: np.ndarray,
    max_val: int = 0,
) -> np.ndarray:
    if max_val == 0:
        max_rows = max(zdl.shape[0], len(ortho), wikt.shape[0], stat.shape[0])
    else:
        max_rows = max(max_val, len(ortho), wikt.shape[0], stat.shape[0])
    try:
        max_cols = max(zdl.shape[1], len(ortho[0]))
    except (IndexError, AttributeError):
        max_cols = max_val or 1

    combined = np.zeros((max_rows, max_cols))
    combined[: zdl.shape[0], : zdl.shape[1]] = zdl
    combined[: len(ortho), : len(ortho[0])] = ortho
    combined[: wikt.shape[0], :1] = wikt.reshape(-1, 1)
    combined[: stat.shape[0], :1] = stat.reshape(-1, 1)
    return combined


def preprocess(text: str, max_val: int = 1931) -> tuple[tf.Tensor, list[tf.Tensor], list]:
    """Extract all features from *text*.

    Returns ``(combined_tensor, zdl_padded, ortho_vector)``.
    """
    logger.info("Analysing text…")
    zdl = _zdl_embedding(text)
    ortho = _ortho_embeddings(text)
    wikt = _wiktionary_embedding(text)
    stat = _stats_embedding(text)

    combined = _concat_features(zdl, ortho, wikt, stat, max_val)
    logger.info("Embedded into shape %s", combined.shape)
    return (
        tf.stack([combined]),
        zero_pad_zdl([zdl], max_val),
        ortho,
    )


# ---------------------------------------------------------------------------
# Main inference function
# ---------------------------------------------------------------------------

TARGETS = ("author_gender", "author_age", "author_education", "author_regiolect")


def predict(text: str) -> dict[str, dict]:
    """Run all pre-trained models on *text* and return a profile dict.

    Returns a dict keyed by attribute name, each value being
    ``{"label": str, "confidence": float}``.
    """
    combined, zdl_padded, ortho = preprocess(text)

    profile: dict[str, dict] = {}
    for target in TARGETS:
        model_path = _find_model(target)
        if model_path is None:
            logger.warning("No trained model found for %s — skipping.", target)
            continue

        model = load_model(str(model_path))

        if target == "author_regiolect":
            pred = model.predict(tf.stack(zdl_padded))
        elif target == "author_education":
            pred = model.predict(tf.stack([ortho]))
        else:
            pred = model.predict(combined)

        confidence = float(max(pred[0]))
        idx = int(np.argmax(np.round(pred), axis=1)[0])
        label = floats_to_labels([idx], target)[0]
        profile[target] = {"label": label, "confidence": confidence}

    return profile


def _find_model(target: str) -> Path | None:
    candidates = [
        TRAINED_MODELS_DIR / f"fully_mapped_features_{target}.model",
        TRAINED_MODELS_DIR / f"ZDL_features_{target}.model",
        TRAINED_MODELS_DIR / f"ortho_features_{target}.model",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Text / path resolution
# ---------------------------------------------------------------------------

_PATH_RE = re.compile(r"""^(?:[a-z]:)?[/\\]{0,2}(?:[./\\ ](?![./\\\n])|[^<>:"|?*./ \\\n])+$""")


def resolve_text_input(input_str: str) -> str:
    """Return text content: read from file if *input_str* looks like a path."""
    if _PATH_RE.search(input_str):
        p = Path(input_str)
        if not p.exists():
            raise ParsedPathNotExistError(
                f"Input looks like a file path but does not exist: {input_str!r}"
            )
        logger.info("Reading file %s…", p)
        return p.read_text(encoding="utf-8")
    return input_str


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

_MALE_ART = """
   ▄████████████████▄
   ██ ██▀▀  ▀▀███ ██
   ██ ██  ▄▄  ███ ██   ♂
   ██ ██████  ███ ██
   ▀████████████████▀
"""
_FEMALE_ART = """
   ▄████████████████▄
   ██ ██▀▀  ▀▀███ ██
   ██ ██  ◡◡  ███ ██  ♀
   ██ ██████  ███ ██
   ▀████████████████▀
"""


def print_profile(profile: dict[str, dict]) -> None:
    """Print a human-readable author profile to stdout."""
    gender_info = profile.get("author_gender")
    edu_info    = profile.get("author_education")
    regio_info  = profile.get("author_regiolect")
    age_info    = profile.get("author_age")

    pronoun_poss, pronoun_subj = ("His", "He") if (gender_info and gender_info["label"] == "male") else ("Her", "She")

    if gender_info:
        g, c = gender_info["label"], gender_info["confidence"]
        print(_MALE_ART if g == "male" else _FEMALE_ART)
        adj = "" if c > 0.9 else "probably "
        print(f"The author is {adj}{g}. (confidence: {c:.1%})")

    if edu_info:
        e, c = edu_info["label"], edu_info["confidence"]
        adj = "" if c > 0.9 else "most likely "
        print(f"{pronoun_poss} education level is {adj}{e}. (confidence: {c:.1%})")

    if regio_info:
        r, c = regio_info["label"], regio_info["confidence"]
        adj = "" if c > 0.9 else "probably "
        print(f"{pronoun_subj} {adj}comes from {r}. (confidence: {c:.1%})")

    if age_info:
        a, c = age_info["label"], age_info["confidence"]
        print(f"{pronoun_subj} is approximately {a} years old. (confidence: {c:.1%})")
