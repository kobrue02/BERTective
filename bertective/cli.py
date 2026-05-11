"""Command-line interface for BERTective.

Usage overview
--------------
  bertective predict <text|path>
  bertective data download [--wiktionary]
  bertective data build [--path data] [--sources REDDIT ACHGUT GUTENBERG]
  bertective features build {zdl,ortho,stats,wikt,all} [--path data]
  bertective train {age,gender,education,regiolect} [options]
  bertective query <label=value>
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from bertective.constants import ABOUT, DEFAULT_SOURCES, LOGO
from bertective.exceptions import CorpusNotFoundError

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------

def cmd_predict(args: argparse.Namespace) -> None:
    import os
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    from bertective.models.predictor import predict, print_profile, resolve_text_input
    text = resolve_text_input(args.text)
    profile = predict(text)
    print_profile(profile)


def cmd_data_download(args: argparse.Namespace) -> None:
    import os
    import scraping_tools.achse_des_guten_api as achse_scraper
    import scraping_tools.korrekturen_scrape as ortho_scraper
    import scraping_tools.plebbit as reddit_scraper
    from scraping_tools.wiktionary_api import download_wiktionary

    path = args.path
    os.makedirs(path, exist_ok=True)
    if args.wiktionary or args.all:
        os.makedirs(f"{path}/wiktionary", exist_ok=True)
        download_wiktionary(path)
    if args.all or not args.wiktionary:
        ortho_scraper.run(path)
        achse_scraper.run(path)
        reddit_scraper.locale_reddits(path)


def cmd_data_build(args: argparse.Namespace) -> None:
    from bertective.data.loaders import build_corpus
    from bertective.constants import CORPUS_AVRO

    data_path = Path(args.path)
    corpus = build_corpus(data_path=data_path, sources=args.sources)
    avro_path = args.output or str(data_path / "corpus.avro")
    corpus.save_to_avro(avro_path)
    print(f"Corpus saved to {avro_path}  ({len(corpus)} items)")


def cmd_features_build(args: argparse.Namespace) -> None:
    from bertective.corpus import DataCorpus
    from bertective.data import loaders

    data_path = Path(args.path)
    corpus = _load_corpus(data_path)

    which = args.feature.lower()

    if which in ("zdl", "all"):
        print("Building ZDL vectors…")
        loaders.build_zdl_vectors(corpus, data_path)
    if which in ("ortho", "all"):
        print("Building orthography matrix…")
        loaders.build_ortho_matrix(corpus)
    if which in ("stats", "all"):
        print("Building statistical feature matrix…")
        loaders.build_stat_matrix(corpus)
    if which in ("wikt", "all"):
        print("Building Wiktionary matrix…")
        loaders.build_wikt_matrix(corpus)
    if which not in ("zdl", "ortho", "stats", "wikt", "all"):
        print(f"Unknown feature type: {which!r}", file=sys.stderr)
        sys.exit(1)


def cmd_train(args: argparse.Namespace) -> None:
    import os; os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    tf.keras.backend.clear_session()

    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    from bertective.models.trainer import run_training, evaluate

    target_map = {
        "age":        "author_age",
        "gender":     "author_gender",
        "education":  "author_education",
        "regiolect":  "author_regiolect",
    }
    target = target_map[args.target]
    data_path = Path(args.path)
    corpus = _load_corpus(data_path)

    print(f"Training {target} model using {args.feature!r} features…")
    model, X_test, y_test = run_training(
        corpus=corpus,
        target=target,
        feature=args.feature,
        sources=args.sources,
        model_type=args.model,
        max_samples=args.max_samples,
        data_path=data_path,
    )
    report, cm = evaluate(model, X_test, y_test, target)
    print(report)

    if args.plot_cm:
        ConfusionMatrixDisplay(cm).plot()
        plt.show()


def cmd_query(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    data_path = Path(args.path)
    corpus = _load_corpus(data_path)

    try:
        label, value = args.query.split("=", 1)
    except ValueError:
        print("Query must be in the form  label=value", file=sys.stderr)
        sys.exit(1)

    results = corpus.query({label: value})
    print(f"Found {len(results)} items where {label}={value!r}")

    if not results:
        return

    dist: dict = {}
    for item in results:
        val = getattr(item, "author_regiolect", "?")
        dist[val] = dist.get(val, 0) + 1

    sns.barplot(x=list(dist.keys()), y=list(dist.values()))
    plt.title(f"Regiolect distribution for {label}={value!r}")
    plt.show()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_corpus(data_path: Path):
    from bertective.corpus import DataCorpus
    avro = data_path / "corpus.avro"
    if not avro.exists():
        raise CorpusNotFoundError(
            f"Corpus file not found: {avro}\n"
            "Run `bertective data build` first."
        )
    corpus = DataCorpus()
    corpus.read_avro(avro)
    print(f"Loaded corpus: {len(corpus)} items")
    return corpus


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bertective",
        description="German author profiling via linguistic features and deep learning.",
    )
    parser.add_argument("--version", action="version", version="bertective 0.1.0")

    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # --- predict ---
    p_predict = sub.add_parser("predict", help="Infer author profile from text or file.")
    p_predict.add_argument("text", help="German text or path to a UTF-8 text file.")
    p_predict.set_defaults(func=cmd_predict)

    # --- data ---
    p_data = sub.add_parser("data", help="Download or build the training corpus.")
    data_sub = p_data.add_subparsers(dest="data_command", metavar="<subcommand>")
    data_sub.required = True

    p_data_dl = data_sub.add_parser("download", help="Download raw data resources.")
    p_data_dl.add_argument("--path", default="data", help="Destination directory.")
    p_data_dl.add_argument("--wiktionary", action="store_true", help="Download Wiktionary word lists only.")
    p_data_dl.add_argument("--all", action="store_true", help="Download all resources.")
    p_data_dl.set_defaults(func=cmd_data_download)

    p_data_build = data_sub.add_parser("build", help="Build corpus AVRO from downloaded data.")
    p_data_build.add_argument("--path", default="data", help="Data root directory.")
    p_data_build.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES,
                              choices=["REDDIT", "ACHGUT", "GUTENBERG"],
                              metavar="SOURCE",
                              help="Data sources to include (default: all).")
    p_data_build.add_argument("--output", default=None,
                              help="Output AVRO path (default: <path>/corpus.avro).")
    p_data_build.set_defaults(func=cmd_data_build)

    # --- features ---
    p_feat = sub.add_parser("features", help="Pre-compute feature vectors for the corpus.")
    feat_sub = p_feat.add_subparsers(dest="feat_command", metavar="<subcommand>")
    feat_sub.required = True

    p_feat_build = feat_sub.add_parser("build", help="Build one or all feature matrices.")
    p_feat_build.add_argument(
        "feature",
        choices=["zdl", "ortho", "stats", "wikt", "all"],
        help=(
            "zdl   — ZDL regional corpus vectors (needed for regiolect)\n"
            "ortho — orthography / spelling-error vectors (needed for education)\n"
            "stats — statistical linguistic features\n"
            "wikt  — Wiktionary vocabulary features\n"
            "all   — build all of the above"
        ),
    )
    p_feat_build.add_argument("--path", default="data", help="Data root directory.")
    p_feat_build.set_defaults(func=cmd_features_build)

    # --- train ---
    p_train = sub.add_parser("train", help="Train a prediction model.")
    p_train.add_argument(
        "target",
        choices=["age", "gender", "education", "regiolect"],
        help="Author attribute to predict.",
    )
    p_train.add_argument(
        "--feature", "-f",
        choices=["zdl", "ortho", "stats", "wikt", "all"],
        default="ortho",
        help="Feature set to use (default: ortho).",
    )
    p_train.add_argument(
        "--model", "-m",
        choices=["multiclass", "rnn", "binary"],
        default="multiclass",
        help="Model architecture (default: multiclass).",
    )
    p_train.add_argument("--path", default="data", help="Data root directory.")
    p_train.add_argument(
        "--sources", nargs="+", default=DEFAULT_SOURCES,
        choices=["REDDIT", "ACHGUT", "GUTENBERG"],
        metavar="SOURCE",
        help="Training data sources.",
    )
    p_train.add_argument(
        "--max-samples", "-n", type=int, default=4000,
        dest="max_samples",
        help="Maximum number of training samples (default: 4000).",
    )
    p_train.add_argument(
        "--plot-cm", action="store_true",
        help="Show confusion matrix after evaluation.",
    )
    p_train.set_defaults(func=cmd_train)

    # --- query ---
    p_query = sub.add_parser("query", help="Query the corpus by label and plot distribution.")
    p_query.add_argument("query", help="Filter expression, e.g. author_age=25")
    p_query.add_argument("--path", default="data", help="Data root directory.")
    p_query.set_defaults(func=cmd_query)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    print(LOGO)
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        args.func(args)
    except CorpusNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nAborted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
