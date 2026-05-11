"""Keras model architectures for author-attribute prediction."""
from __future__ import annotations

from keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LSTM,
    MaxPooling1D,
    TimeDistributed,
)
from keras.losses import SparseCategoricalCrossentropy
from keras.models import Sequential
from keras.optimizers.legacy import RMSprop
from keras.optimizers import SGD


def build_multiclass_model(input_shape: tuple, n_classes: int) -> Sequential:
    """Dense feed-forward classifier for multiclass prediction.

    Architecture: Flatten → Dense(128) → Dropout → Dense(64) → Dropout
                  → Dense(32) → Dropout → Dense(n_classes, softmax)
    """
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(64, activation="relu"),
        Dropout(0.25),
        Dense(32, activation="relu"),
        Dropout(0.25),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0),
        metrics=["accuracy"],
    )
    return model


def build_binary_model(n_inputs: int) -> Sequential:
    """Binary classifier (gender prediction)."""
    model = Sequential([
        Flatten(input_shape=(n_inputs,)),
        Dense(n_inputs, activation="relu"),
        Dropout(0.5),
        Dense(16, activation="relu"),
        Dropout(0.5),
        Dense(1, activation="softmax"),
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer=SGD(learning_rate=0.01),
        metrics=["accuracy"],
    )
    return model


def build_rnn_model(input_shape: tuple, n_classes: int) -> Sequential:
    """Bidirectional LSTM with a Conv1D front-end, suited for ZDL sequence vectors.

    :param input_shape: e.g. ``(1, max_doc_len, 6)`` for ZDL inputs.
    :param n_classes: number of output classes.
    """
    model = Sequential([
        Input(shape=input_shape),
        TimeDistributed(Conv1D(32, 4, activation="relu", data_format="channels_last")),
        TimeDistributed(MaxPooling1D(2)),
        TimeDistributed(Flatten()),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.25),
        Bidirectional(LSTM(32)),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        loss=SparseCategoricalCrossentropy(from_logits=True),
        optimizer=RMSprop(learning_rate=0.001, rho=0.9, epsilon=None, decay=0.0),
        metrics=["accuracy"],
    )
    return model
