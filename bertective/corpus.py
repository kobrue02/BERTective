"""DataObject and DataCorpus — core data containers for BERTective."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, overload

import pandas as pd
from fastavro import parse_schema, reader, writer

from bertective.constants import VALID_EDUCATION, VALID_REGIOLECTS
from bertective.exceptions import InvalidLabelError

logger = logging.getLogger(__name__)

_AVRO_SCHEMA: dict = {
    "doc": "corpus",
    "name": "corpus",
    "namespace": "corpus",
    "type": "record",
    "fields": [
        {"name": "text",             "type": "string"},
        {"name": "author_age",       "type": "int"},
        {"name": "author_gender",    "type": "string"},
        {"name": "author_regiolect", "type": "string"},
        {"name": "author_education", "type": "string"},
        {"name": "source",           "type": "string"},
    ],
}


@dataclass
class DataObject:
    """An annotated text sample with optional author metadata.

    Sentinel values ("NONE" / 0) indicate missing metadata.
    ``id`` is assigned by DataCorpus on insertion and should not be set manually.
    """

    text: str
    author_age: int = 0
    author_gender: str = "NONE"
    author_regiolect: str = "NONE"
    author_education: str = "NONE"
    source: str = "NONE"
    id: int = field(default=-1, compare=False, repr=False)

    # ------------------------------------------------------------------
    # Backwards-compatible dict-style access
    # ------------------------------------------------------------------

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value) -> None:
        setattr(self, key, value)

    @property
    def content(self) -> "DataObject":
        """Shim so legacy ``item.content['id']`` calls still resolve."""
        return self

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def as_dict(self) -> dict:
        return {
            "text":             self.text,
            "author_age":       self.author_age,
            "author_gender":    self.author_gender,
            "author_regiolect": self.author_regiolect,
            "author_education": self.author_education,
            "source":           self.source,
        }


class DataCorpus:
    """An ordered, appendable collection of annotated text samples.

    Items are accessed by 0-based index or slice.  Each item receives a
    unique sequential ``id`` on insertion, which is used as a key in the
    pre-computed vector stores.
    """

    def __init__(self) -> None:
        self._items: list[DataObject] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_item(self, item: DataObject) -> None:
        """Validate *item*, assign the next sequential ID, and append it."""
        _validate(item)
        item.id = len(self._items)
        self._items.append(item)

    def from_list(self, items: list[DataObject]) -> None:
        """Replace corpus contents with *items* (IDs are re-assigned from 0)."""
        self._items = []
        for item in items:
            self.add_item(item)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, search: dict[str, str]) -> list[DataObject]:
        """Return all items where ``getattr(item, label) == value``."""
        label, value = next(iter(search.items()))
        return [item for item in self._items if getattr(item, label, None) == value]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([item.as_dict() for item in self._items])

    def save_to_avro(self, path: str | Path) -> None:
        records = [item.as_dict() for item in self._items]
        parsed = parse_schema(_AVRO_SCHEMA)
        with open(path, "wb") as fh:
            writer(fh, parsed, records)
        logger.info("Corpus saved → %s  (%d items)", path, len(self._items))

    def read_avro(self, path: str | Path) -> None:
        """Append all records from *path* into this corpus."""
        with open(path, "rb") as fh:
            for record in reader(fh):
                obj = DataObject(
                    text=record["text"],
                    author_age=record["author_age"],
                    author_gender=record["author_gender"],
                    author_regiolect=record["author_regiolect"],
                    author_education=record["author_education"],
                    source=record["source"],
                )
                self.add_item(obj)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    @overload
    def __getitem__(self, i: int) -> DataObject: ...
    @overload
    def __getitem__(self, i: slice) -> list[DataObject]: ...

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[DataObject]:
        return iter(self._items)

    # legacy attribute name used by old code
    @property
    def corpus(self) -> list[DataObject]:
        return self._items


# ---------------------------------------------------------------------------
# Validation (module-level to keep DataCorpus lean)
# ---------------------------------------------------------------------------

def _validate(item: DataObject) -> None:
    if not isinstance(item.text, str):
        raise InvalidLabelError("text must be a string")
    if not isinstance(item.author_age, int) or len(str(abs(item.author_age))) > 3:
        raise InvalidLabelError(f"author_age={item.author_age!r} is not a valid age")
    if item.author_regiolect not in VALID_REGIOLECTS:
        raise InvalidLabelError(
            f"author_regiolect={item.author_regiolect!r} must be one of {sorted(VALID_REGIOLECTS)}"
        )
    if item.author_education not in VALID_EDUCATION:
        raise InvalidLabelError(
            f"author_education={item.author_education!r} must be one of {sorted(VALID_EDUCATION)}"
        )
