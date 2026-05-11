"""Data source loaders: Reddit, Achse des Guten, Project Gutenberg."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from bertective.constants import AREAL_DICT
from bertective.corpus import DataCorpus, DataObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language detection helper
# ---------------------------------------------------------------------------

def _is_german(text: str) -> bool:
    from langdetect import detect, DetectorFactory, lang_detect_exception
    DetectorFactory.seed = 0
    if text in ("[removed]", "[deleted]") or len(text) < 20 or len(text.split()) < 3:
        return False
    try:
        return detect(text) == "de"
    except lang_detect_exception.LangDetectException:
        return False


# ---------------------------------------------------------------------------
# Source-specific loaders
# ---------------------------------------------------------------------------

def _load_reddit_locales(corpus: DataCorpus, data_path: Path) -> None:
    """Load localised Reddit posts and annotate with regiolect from AREAL_DICT."""
    locale_dir = data_path / "reddit" / "locales"
    if not locale_dir.exists():
        logger.warning("Reddit locales directory not found: %s", locale_dir)
        return

    city_to_areal = {city: areal for areal, cities in AREAL_DICT.items() for city in cities}

    for fname in tqdm(sorted(locale_dir.iterdir()), desc="Reddit locales"):
        if fname.suffix != ".json":
            continue
        city = fname.stem
        areal = city_to_areal.get(city)
        if areal is None:
            continue
        with open(fname, encoding="utf-8") as fh:
            file_data = json.load(fh)
        for item in file_data.get("data", []):
            text = item.get("selftext", "")
            if _is_german(text):
                corpus.add_item(DataObject(text=text, author_regiolect=areal, source="REDDIT"))


def _load_achgut(corpus: DataCorpus, data_path: Path) -> None:
    """Load annotated Achse des Guten blog posts."""
    parquet = data_path / "achse" / "achse_des_guten_annotated_items.parquet"
    if not parquet.exists():
        logger.warning("Achse des Guten parquet not found: %s", parquet)
        return

    df = pd.read_parquet(parquet)
    for row in df.itertuples(index=False):
        corpus.add_item(DataObject(
            text=row.content,
            author_age=row.age,
            author_gender=row.sex,
            author_education=row.education,
            author_regiolect=row.regiolect,
            source="ACHGUT",
        ))


def _load_reddit_annotated(corpus: DataCorpus, data_path: Path) -> None:
    """Load annotated Reddit posts from the consolidated parquet file."""
    parquet = data_path / "reddit" / "annotated_posts_2.parquet"
    if not parquet.exists():
        logger.warning("Annotated Reddit parquet not found: %s", parquet)
        return

    df = pd.read_parquet(parquet)
    for row in df.itertuples(index=False):
        regiolect = getattr(row, "regiolect", "NONE") or "NONE"
        corpus.add_item(DataObject(
            text=row.content,
            author_age=row.age,
            author_gender=row.sex,
            author_regiolect=regiolect,
            source="REDDIT",
        ))


def _load_gutenberg(corpus: DataCorpus, data_path: Path) -> None:
    """Load Project Gutenberg texts with author metadata."""
    data_file = data_path / "gutenberg" / "data.json"
    author_file = data_path / "gutenberg" / "author_dict.json"
    if not data_file.exists() or not author_file.exists():
        logger.warning("Gutenberg data files not found in %s", data_path / "gutenberg")
        return

    with open(data_file) as fh:
        data = json.load(fh)
    with open(author_file) as fh:
        author_dict = json.load(fh)

    author_index = {book["book_name"]: book for book in author_dict.get("books", [])}

    for entry in tqdm(data.get("texts", []), desc="Gutenberg"):
        meta = author_index.get(entry.get("book_name", ""))
        if meta is None or not meta.get("author_age"):
            continue
        try:
            corpus.add_item(DataObject(
                text=entry["text"],
                author_age=int(meta["author_age"]),
                author_gender=meta.get("author_gender", "NONE"),
                source="GUTENBERG",
            ))
        except (ValueError, KeyError):
            continue


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

_LOADERS = {
    "REDDIT":    (_load_reddit_locales, _load_reddit_annotated),
    "ACHGUT":    (_load_achgut,),
    "GUTENBERG": (_load_gutenberg,),
}


def build_corpus(
    data_path: Path = Path("data"),
    sources: list[str] | None = None,
) -> DataCorpus:
    """Build a DataCorpus from all enabled data sources.

    :param data_path: Root directory containing the data sub-folders.
    :param sources: Which sources to include; defaults to all (REDDIT, ACHGUT, GUTENBERG).
    """
    if sources is None:
        sources = ["REDDIT", "ACHGUT", "GUTENBERG"]

    corpus = DataCorpus()
    for source in sources:
        loaders = _LOADERS.get(source.upper(), ())
        for loader in loaders:
            loader(corpus, data_path)
        logger.info("Loaded %s → corpus now has %d items", source, len(corpus))

    return corpus


# ---------------------------------------------------------------------------
# Feature-vector builders (delegated to feature modules)
# ---------------------------------------------------------------------------

def build_zdl_vectors(corpus: DataCorpus, data_path: Path = Path("data")) -> None:
    """Vectorise corpus with ZDL regional embeddings and write batched parquets."""
    from bertective.features.zdl import ZDLVectorMatrix

    batch_size = 1000
    n = len(corpus)
    zdl_dir = data_path / "ZDL"
    zdl_dir.mkdir(parents=True, exist_ok=True)

    for k in range(n // batch_size + 1):
        out = zdl_dir / f"zdl_word_embeddings_batch_{k}.parquet"
        if out.exists():
            logger.info("Batch %d already done, skipping.", k)
            continue

        start = k * batch_size
        end = min((k + 1) * batch_size, n - 1)
        tqdm.write(f"Vectorising batch {k + 1}…")

        matrix = ZDLVectorMatrix(source=corpus[start:end]).vectors
        for key in matrix:
            matrix[key] = matrix[key].tolist()

        import pandas as pd
        pd.DataFrame(matrix.items(), columns=["ID", "embedding"]).to_parquet(out)


def build_ortho_matrix(corpus: DataCorpus, output: Path = Path("vectors/orthography_matrix.json")) -> None:
    """Compute orthography vectors for all corpus items and save as JSON."""
    from bertective.features.ortho import OrthoMatrixModel
    import json

    ortho = OrthoMatrixModel()
    result = {str(item.id): ortho.embed(item.text) for item in tqdm(corpus, desc="Ortho")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    logger.info("Orthography matrix saved → %s", output)


def build_stat_matrix(corpus: DataCorpus, output: Path = Path("vectors/statistical_matrix.json")) -> None:
    """Compute statistical features for all corpus items and save as JSON."""
    from bertective.features.stats import Statistext
    import json

    result = {str(item.id): list(Statistext(item.text).all_stats) for item in tqdm(corpus, desc="Stats")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        json.dump(result, fh)
    logger.info("Statistical matrix saved → %s", output)


def build_wikt_matrix(corpus: DataCorpus, output: Path = Path("data/wiktionary/wiktionary.parquet")) -> None:
    """Compute Wiktionary feature vectors for all corpus items and save as parquet."""
    from bertective.features.wiktionary import WiktionaryModel

    model = WiktionaryModel(source=corpus)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.df_matrix.to_parquet(output)
    logger.info("Wiktionary matrix saved → %s", output)
