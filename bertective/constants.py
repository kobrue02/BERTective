"""Project-wide constants: label mappings, region maps, and file paths."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# --- Directories ---
VECTORS_DIR = PROJECT_ROOT / "vectors"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
TRAINED_MODELS_DIR = MODELS_DIR / "trained_models"
ZDL_DATA_DIR = DATA_DIR / "ZDL"

# --- Resource files ---
ZDL_VECTOR_DICT = VECTORS_DIR / "zdl_vector_dict.json"
ORTHO_MATRIX = VECTORS_DIR / "orthography_matrix.json"
STAT_MATRIX = VECTORS_DIR / "statistical_matrix.json"
WIKTIONARY_JSON = DATA_DIR / "wiktionary" / "wiktionary.json"
WIKTIONARY_PARQUET = DATA_DIR / "wiktionary" / "wiktionary.parquet"
WIKTIONARY_PICKLE = VECTORS_DIR / "wiktionary.pickle"
EMOTICONS_FILE = MODELS_DIR / "wikipedia_emoticons.txt"
ORTHOGRAPHY_JSON = DATA_DIR / "annotation" / "orthography.json"
ERROR_TUPLES_JSON = DATA_DIR / "annotation" / "error_tuples.json"
STOPWORDS_CACHE = VECTORS_DIR / "german_stopwords.txt"
CORPUS_AVRO = DATA_DIR / "corpus.avro"

# --- Valid label sets ---
VALID_REGIOLECTS: frozenset[str] = frozenset({
    "DE-NORTH-WEST", "DE-NORTH-EAST",
    "DE-MIDDLE-WEST", "DE-MIDDLE-EAST",
    "DE-SOUTH-WEST", "DE-SOUTH-EAST",
    "NONE", "",
})
VALID_EDUCATION: frozenset[str] = frozenset({
    "finished_highschool", "in_university", "has_bachelor",
    "has_master", "has_phd", "is_apprentice", "has_apprentice",
    "NONE", "",
})

# --- Label ↔ integer mappings ---
REGIOLECT_TO_INT: dict[str, float] = {
    "DE-MIDDLE-EAST": 0.0, "DE-MIDDLE-WEST": 1.0,
    "DE-NORTH-EAST":  2.0, "DE-NORTH-WEST":  3.0,
    "DE-SOUTH-EAST":  4.0, "DE-SOUTH-WEST":  5.0,
}
INT_TO_REGIOLECT: dict[float, str] = {v: k for k, v in REGIOLECT_TO_INT.items()}

EDUCATION_TO_INT: dict[str, float] = {
    "finished_highschool": 0.0,
    "has_phd":             1.0,
    "has_apprentice":      2.0,
    "has_master":          3.0,
}
INT_TO_EDUCATION: dict[float, str] = {v: k for k, v in EDUCATION_TO_INT.items()}

GENDER_TO_INT: dict[str, float] = {
    "female": 0.0, "f": 0.0,
    "male":   1.0, "m": 1.0,
}
INT_TO_GENDER: dict[float, str] = {0.0: "female", 1.0: "male"}

INT_TO_AGE: dict[float, str] = {
    0.0: "10-25", 1.0: "25-40", 2.0: "40-60", 3.0: "60+",
}

LABEL_MAPS: dict[str, dict[str, float]] = {
    "author_regiolect": REGIOLECT_TO_INT,
    "author_education": EDUCATION_TO_INT,
    "author_gender":    GENDER_TO_INT,
}
REVERSE_LABEL_MAPS: dict[str, dict[float, str]] = {
    "author_regiolect": INT_TO_REGIOLECT,
    "author_education": INT_TO_EDUCATION,
    "author_gender":    INT_TO_GENDER,
    "author_age":       INT_TO_AGE,
}

# --- German areal regions: city name → areal ---
AREAL_DICT: dict[str, list[str]] = {
    "DE-NORTH-EAST": ["Rostock", "Berlin", "luebeck", "Potsdam"],
    "DE-NORTH-WEST": ["bremen", "Hamburg", "Hannover", "bielefeld", "Dortmund", "kiel", "Paderborn"],
    "DE-MIDDLE-EAST": ["Leipzig", "dresden", "HalleSaale", "erfurt", "jena"],
    "DE-MIDDLE-WEST": [
        "frankfurt", "duesseldorf", "cologne", "Bonn", "Mainz",
        "Wiesbaden", "kaiserslautern", "Mannheim", "Saarland",
    ],
    "DE-SOUTH-EAST": ["Munich", "Nurnberg", "Regensburg", "Würzburg", "bayreuth", "bavaria"],
    "DE-SOUTH-WEST": [
        "stuttgart", "augsburg", "freiburg", "karlsruhe",
        "Ulm", "Tuebingen", "Ludwigsburg", "Heidelberg",
    ],
}

# --- Default data sources ---
DEFAULT_SOURCES: list[str] = ["ACHGUT", "REDDIT", "GUTENBERG"]

# --- Logo / about text ---
LOGO = r"""
 /$$$$$$$  /$$$$$$$$ /$$$$$$$  /$$$$$$$$                    /$$     /$$
| $$__  $$| $$_____/| $$__  $$|__  $$__/                   | $$    |__/
| $$  \ $$| $$      | $$  \ $$   | $$  /$$$$$$   /$$$$$$$ /$$$$$$   /$$
| $$$$$$$ | $$$$$   | $$$$$$$/   | $$ /$$__  $$ /$$_____/|_  $$_/  | $$
| $$__  $$| $$__/   | $$__  $$   | $$| $$$$$$$$| $$        | $$    | $$
| $$  \ $$| $$      | $$  \ $$   | $$| $$_____/| $$        | $$ /$$| $$
| $$$$$$$/| $$$$$$$$| $$  | $$   | $$|  $$$$$$$|  $$$$$$$  |  $$$$/| $$
|_______/ |________/|__/  |__/   |__/ \_______/ \_______/   \___/  |__/
"""

ABOUT = (
    "BERTective is a German author profiling tool trained on multiple data sources "
    "using linguistic features (ZDL regional corpus, Wiktionary, orthography, statistics) "
    "and deep learning.\n\n"
    "Run `bertective --help` to see available commands."
)
