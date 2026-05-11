"""Model training pipeline."""
from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

from bertective.constants import (
    AREAL_DICT,
    INT_TO_AGE,
    INT_TO_GENDER,
    INT_TO_REGIOLECT,
    INT_TO_EDUCATION,
    LABEL_MAPS,
    ORTHO_MATRIX,
    STAT_MATRIX,
    TRAINED_MODELS_DIR,
    ZDL_DATA_DIR,
)
from bertective.corpus import DataCorpus
from bertective.exceptions import MissingTrainingData
from bertective.features.wiktionary import WiktionaryModel
from bertective.models.architectures import (
    build_binary_model,
    build_multiclass_model,
    build_rnn_model,
)

logger = logging.getLogger(__name__)

REVERSE_MAPS: dict[str, dict] = {
    "author_regiolect": INT_TO_REGIOLECT,
    "author_education": INT_TO_EDUCATION,
    "author_gender":    INT_TO_GENDER,
    "author_age":       INT_TO_AGE,
}

_SENTINEL = {"N/A", "NONE", "", 0, "0", None}


# ---------------------------------------------------------------------------
# Label conversion helpers
# ---------------------------------------------------------------------------

def labels_to_float(labels: list, target: str) -> list[float]:
    """Convert string/int labels to floats for model training."""
    if target == "author_age":
        return [_age_bin(int(a)) for a in labels]
    mapping = LABEL_MAPS[target]
    return [mapping[l] for l in labels]


def floats_to_labels(values: list, target: str) -> list[str]:
    """Convert numeric predictions back to human-readable strings."""
    mapping = REVERSE_MAPS[target]
    if target == "author_gender":
        values = ["female" if str(v) in ("f", "female") else "male" if str(v) in ("m", "male") else v for v in values]
    if isinstance(values[0], (int, np.integer)) and target == "author_age":
        return [_age_label(v) for v in values]
    return [mapping[float(v)] for v in values]


def _age_bin(age: int) -> float:
    if age <= 25:
        return 0.0
    if age <= 40:
        return 1.0
    if age <= 60:
        return 2.0
    return 3.0


def _age_label(age: int) -> str:
    if age <= 25:
        return "10-25"
    if age <= 40:
        return "25-40"
    if age <= 60:
        return "40-60"
    return "60+"


# ---------------------------------------------------------------------------
# Zero-padding helpers
# ---------------------------------------------------------------------------

def zero_pad_zdl(vectors: list, max_len: int) -> list[tf.Tensor]:
    """Pad ZDL vectors to *max_len* along the sequence dimension."""
    result = []
    for v in vectors:
        t = tf.convert_to_tensor(v, tf.float64)
        if len(t.shape) == 3:
            diff = max_len - t.shape[1]
            t = tf.pad(t, [[0, 0], [0, diff], [0, 0]])
        else:
            t = tf.zeros([1, max_len, 6], tf.float64)
        result.append(t)
    return result


def _maxval(embeddings: list) -> int:
    best = 0
    for x in embeddings:
        x = np.array(x)
        if x.ndim > 1 and x.shape[0] > best:
            best = x.shape[0]
    return best


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def _read_zdl_parquet(path: Path) -> Any:
    import pandas as pd
    frames = []
    for f in tqdm(sorted(path.iterdir())):
        if f.suffix == ".parquet":
            frames.append(pd.read_parquet(f))
    return pd.concat(frames)


def load_features(
    corpus: DataCorpus,
    target: str,
    feature: str,
    ids: list[int],
    data_path: Path,
) -> tuple[Any, str, Any, int]:
    """Load pre-computed feature vectors for the given *ids*.

    Returns ``(X, source_name, raw_vectors, max_sequence_length)``.
    """
    max_val = 1931  # default ZDL pad target

    if feature.lower() == "ortho":
        if not ORTHO_MATRIX.exists():
            raise MissingTrainingData(
                f"Orthography matrix not found at {ORTHO_MATRIX}. Run: bertective features build ortho"
            )
        with open(ORTHO_MATRIX) as fh:
            vectors = json.load(fh)
        X = [np.array(list(vectors[str(i)].values())) for i in ids]
        return X, "ortho", vectors, max_val

    if feature.lower() == "stat":
        if not STAT_MATRIX.exists():
            raise MissingTrainingData(
                f"Statistical matrix not found at {STAT_MATRIX}. Run: bertective features build stats"
            )
        with open(STAT_MATRIX) as fh:
            vectors = json.load(fh)
        X = [np.array(vectors[str(i)]) for i in ids]
        return X, "stat", vectors, max_val

    if feature.lower() == "zdl":
        zdl_dir = data_path / "ZDL"
        if not zdl_dir.exists():
            raise MissingTrainingData(
                f"ZDL vectors not found at {zdl_dir}. Run: bertective features build zdl"
            )
        df = _read_zdl_parquet(zdl_dir)
        max_val = _maxval([np.array(list(e)) for e in df.embedding])
        X = [df[df["ID"] == i].embedding.tolist() for i in ids]
        return X, "zdl", df, max_val

    if feature.lower() == "wikt":
        from bertective.constants import WIKTIONARY_PARQUET
        if not WIKTIONARY_PARQUET.exists():
            raise MissingTrainingData(
                f"Wiktionary matrix not found. Run: bertective features build wikt"
            )
        model = WiktionaryModel(path=WIKTIONARY_PARQUET)
        X = [model[i] for i in ids]
        return X, "wikt", model, max_val

    if feature.lower() == "all":
        X, max_val = _concat_all_features(corpus, ids, data_path)
        return X, "all", None, max_val

    raise ValueError(f"Unknown feature type: {feature!r}")


def _concat_all_features(
    corpus: DataCorpus,
    ids: list[int],
    data_path: Path,
) -> tuple[dict[int, np.ndarray], int]:
    from bertective.constants import WIKTIONARY_PARQUET

    zdl_df = _read_zdl_parquet(data_path / "ZDL")
    max_val = _maxval([np.array(list(e)) for e in zdl_df.embedding])

    with open(STAT_MATRIX) as fh:
        stats = json.load(fh)
    with open(ORTHO_MATRIX) as fh:
        ortho = json.load(fh)
    wikt = WiktionaryModel(path=WIKTIONARY_PARQUET)

    features: dict[int, np.ndarray] = {}
    for i in tqdm(ids, desc="Concatenating features"):
        try:
            zdl = np.array(list(zdl_df.loc[zdl_df["ID"] == i, "embedding"].iloc[0]))
            if zdl.size == 0:
                zdl = np.zeros((max_val, 6))
        except IndexError:
            zdl = np.zeros((max_val, 6))

        ortho_v = np.array(list(ortho[str(i)].values()))
        wikt_v = wikt[i]
        stat_v = np.array(list(stats[str(i)]))

        max_rows = max(max_val, len(ortho_v), wikt_v.shape[0], stat_v.shape[0])
        try:
            max_cols = max(zdl.shape[1], len(ortho_v[0]))
        except (IndexError, AttributeError):
            max_cols = max_val

        combined = np.zeros((max_rows, max_cols))
        combined[: zdl.shape[0], : zdl.shape[1]] = zdl
        combined[: len(ortho_v), : len(ortho_v[0])] = ortho_v
        combined[: wikt_v.shape[0], :1] = wikt_v.reshape(-1, 1)
        combined[: stat_v.shape[0], :1] = stat_v.reshape(-1, 1)
        features[i] = combined

    return features, max_val


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def run_training(
    corpus: DataCorpus,
    target: str,
    feature: str,
    sources: list[str],
    model_type: str = "multiclass",
    max_samples: int = 4000,
    data_path: Path = Path("data"),
) -> tuple[Any, Any, list]:
    """Train a model and return ``(model, X_test, y_test)``."""

    # Collect eligible IDs
    ids = [
        item.id
        for item in corpus
        if item.source in sources and item[target] not in _SENTINEL
    ]
    if max_samples < len(ids):
        ids = random.sample(ids, max_samples)

    labels = [corpus[i][target] for i in ids]

    logger.info("Label distribution: %s", {l: labels.count(l) for l in set(labels)})

    X_data, source, vectors, max_val = load_features(corpus, target, feature, ids, data_path)

    ids_shuffled, y_shuffled = shuffle(ids, labels, random_state=3)
    ids_train, ids_test, y_train_raw, y_test_raw = train_test_split(
        ids_shuffled, y_shuffled, test_size=0.2, random_state=42
    )

    y_train = tf.stack(labels_to_float(y_train_raw, target))
    y_test_num = labels_to_float(y_test_raw, target)

    model, X_test, y_test = _fit(
        X=X_data,
        vectors=vectors,
        ids_train=ids_train,
        ids_test=ids_test,
        y_train=y_train,
        y_test=y_test_num,
        source=source,
        model_type=model_type,
        max_val=max_val,
        target=target,
    )
    return model, X_test, y_test_raw


def _fit(
    X, vectors, ids_train, ids_test, y_train, y_test,
    source, model_type, max_val, target,
):
    y_test_t = tf.stack(y_test)
    early_stop = EarlyStopping(monitor="val_loss", patience=8)
    TRAINED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_builders = {
        "multiclass": build_multiclass_model,
        "rnn": build_rnn_model,
        "binary": build_binary_model,
    }
    build = model_builders[model_type]

    if source == "zdl":
        X_train = tf.stack(zero_pad_zdl([vectors[vectors["ID"] == i].embedding.tolist() for i in ids_train], max_val))
        X_test  = tf.stack(zero_pad_zdl([vectors[vectors["ID"] == i].embedding.tolist() for i in ids_test],  max_val))
        model = build((1, max_val, 6), y_train.shape[0])
        _train_loop(model, X_train, X_test, y_train, y_test_t, model_type, early_stop)
        model.save(TRAINED_MODELS_DIR / f"ZDL_features_{target}.model")

    elif source == "wikt":
        X_train = tf.stack([vectors[i] for i in ids_train])
        X_test  = tf.stack([vectors[i] for i in ids_test])
        model = build((27,), y_train.shape[0])
        _train_loop(model, X_train, X_test, y_train, y_test_t, model_type, early_stop)

    elif source == "ortho":
        X_train = tf.stack([list(vectors[str(i)].values()) for i in ids_train])
        X_test  = tf.stack([list(vectors[str(i)].values()) for i in ids_test])
        model = build((5, 96), y_train.shape[0])
        _train_loop(model, X_train, X_test, y_train, y_test_t, model_type, early_stop)
        model.save(TRAINED_MODELS_DIR / f"ortho_features_{target}.model")

    elif source == "stat":
        X_train = tf.stack([np.array(vectors[str(i)]) for i in ids_train])
        X_test  = tf.stack([np.array(vectors[str(i)]) for i in ids_test])
        model = build((14,), y_train.shape[0])
        _train_loop(model, X_train, X_test, y_train, y_test_t, model_type, early_stop)

    elif source == "all":
        X_train = tf.stack([X[i] for i in ids_train])
        X_test  = tf.stack([X[i] for i in ids_test])
        model = build((max_val, 96), y_train.shape[0])
        _train_loop(model, X_train, X_test, y_train, y_test_t, model_type, early_stop)
        model.save(TRAINED_MODELS_DIR / f"fully_mapped_features_{target}.model")

    return model, X_test, list(y_test)


def _train_loop(model, X_train, X_test, y_train, y_test, model_type, early_stop):
    batch = 128 if model_type == "rnn" else 64
    epochs = 128
    try:
        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch,
            validation_data=(X_test, y_test),
            use_multiprocessing=True,
            workers=os.cpu_count() or 4,
            callbacks=[early_stop],
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test_raw: list, target: str) -> tuple[str, np.ndarray]:
    """Run model inference on *X_test* and return a classification report."""
    y_pred = model.predict(X_test)
    y_pred_labels = floats_to_labels(np.argmax(np.round(y_pred), axis=1), target)
    y_true_labels = floats_to_labels(y_test_raw, target)
    report = classification_report(y_true_labels, y_pred_labels)
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    return report, cm
