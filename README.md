# BERTective

BERTective is a German-language author profiling tool. Given a text sample it predicts author attributes — age, gender, regional dialect (Regiolekt), and education level — using a combination of linguistic feature vectors and deep learning.

## Installation

```bash
git clone https://github.com/kobrue02/BERTective.git
cd BERTective
uv sync          # installs all dependencies into a virtual environment
```

Run any command via `uv run bertective …`, or activate the environment first (`source .venv/bin/activate`) and use `bertective …` directly.

---

## Usage

### Predict author profile

```bash
bertective predict "Dies ist ein Text, dessen Autor leider unbekannt ist."
```

Pass a file path instead of inline text:

```bash
bertective predict path/to/text.txt
```

---

### Data — download & build corpus

Download all training data and Wiktionary word lists:

```bash
bertective data download --all
```

Download only the Wiktionary word lists:

```bash
bertective data download --wiktionary
```

Build the corpus AVRO from downloaded data (all sources by default):

```bash
bertective data build
```

Restrict to specific sources:

```bash
bertective data build --sources REDDIT ACHGUT
```

---

### Features — pre-compute vector matrices

Build a single feature matrix:

```bash
bertective features build zdl      # ZDL regional corpus vectors  (for regiolect)
bertective features build ortho    # orthography / spelling-error vectors  (for education)
bertective features build stats    # statistical linguistic features
bertective features build wikt     # Wiktionary vocabulary vectors
```

Build all matrices at once:

```bash
bertective features build all
```

---

### Train a model

```bash
bertective train age       --feature all
bertective train gender    --feature all
bertective train education --feature ortho
bertective train regiolect --feature zdl --model rnn --max-samples 20000
```

| Flag | Options | Default |
|---|---|---|
| `--feature`, `-f` | `zdl`, `ortho`, `stats`, `wikt`, `all` | `ortho` |
| `--model`, `-m` | `multiclass`, `rnn`, `binary` | `multiclass` |
| `--max-samples`, `-n` | integer | `4000` |
| `--sources` | `REDDIT`, `ACHGUT`, `GUTENBERG` | all three |
| `--plot-cm` | flag | off |

---

### Query the corpus

```bash
bertective query "author_regiolect=DE-NORTH-WEST"
```

---

## Architecture

![architecture](https://github.com/kobrue02/BERTective/blob/main/architecture.drawio.svg)

### Feature types

| Feature | Dimensionality | Used for |
|---|---|---|
| ZDL regional vectors | `(seq_len, 6)` per text | Regiolect |
| Wiktionary vocabulary | `(27,)` | General |
| Orthography / error | `(5, 96)` | Education |
| Statistical (spaCy) | `(20,)` | General |

### Project layout

```
bertective/          # core package
  cli.py             # entry point
  corpus.py          # DataObject, DataCorpus
  constants.py       # label maps, file paths, region dict
  features/          # ZDL, Wiktionary, ortho, stats extractors
  models/            # Keras architectures, trainer, predictor
  data/              # corpus builders and feature matrix builders
scraping_tools/      # data collection scripts
data/                # training data (not committed)
vectors/             # pre-computed feature matrices (not committed)
models/              # saved Keras models (not committed)
```
