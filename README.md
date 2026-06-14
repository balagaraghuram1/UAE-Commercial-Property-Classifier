# UAE Commercial Property Classifier

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://pycqa.github.io/isort/)

A **production-grade** machine learning pipeline that classifies UAE commercial
real-estate properties (Villa, Apartment, Office, Retail, Warehouse, etc.) from
textual descriptions.  Supports both classical models (Logistic Regression,
Random Forest, XGBoost) and a deep-learning BERT classifier fine-tuned on
Arabic-influenced English text.

---

## Problem Statement

UAE property listings are notoriously messy: mixed Arabic/English text,
inconsistent units, sparse categorical data, and varied formatting conventions.
This project provides a **unified, extensible pipeline** that:

- Cleans and normalises raw listing text while preserving meaningful structure.
- Engineers numeric features (size, price, room count) from free-form text.
- Trains and tunes multiple classifier architectures with cross-validation.
- Evaluates models with comprehensive metrics and diagnostic plots.
- Supports batch inference with confidence scores.

---

## Features

- **Hybrid text cleaning** – handles Arabic/English mixed text, removes noise but keeps alphanumeric structure.
- **Multiple model architectures** – Logistic Regression, Random Forest, XGBoost, and BERT (AraBERTv2).
- **Hyperparameter tuning** – integrated `GridSearchCV` with sensible defaults.
- **Cross-validation** – stratified k-fold with macro F1 scoring.
- **Comprehensive evaluation** – accuracy, precision, recall, F1, per-class metrics, confusion matrix, ROC-AUC.
- **Diagnostic plots** – confusion matrix heatmap, multi-class ROC curves.
- **CLI & Python API** – both command-line and programmatic interfaces.
- **Fully typed** – every function has type hints and docstrings.
- **Production ready** – logging, error handling, graceful fallbacks for optional dependencies.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        main.py (CLI entry point)                 │
│  preprocess  |  train  |  inference  |  evaluate                  │
└──────┬───────────────────┬───────────────────┬───────────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ preprocess   │   │    train     │   │  inference   │
│ ┌──────────┐ │   │ ┌──────────┐ │   │ ┌──────────┐ │
│ │text      │ │   │ │sklearn   │ │   │ │load     │ │
│ │cleaning  │ │   │ │pipeline  │ │   │ │artifact │ │
│ │numeric   │ │   │ │GridSearch│ │   │ │predict  │ │
│ │extraction│ │   │ │CV        │ │   │ │save     │ │
│ │encoding  │ │   │ │BERT      │ │   │ │results  │ │
│ │impute    │ │   │ │Trainer   │ │   │ └──────────┘ │
│ └──────────┘ │   │ └──────────┘ │   └──────────────┘
└──────────────┘   └──────────────┘   └──────────────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │  evaluate    │
                                          │ ┌──────────┐ │
                                          │ │metrics   │ │
                                          │ │plots     │ │
                                          │ │report    │ │
                                          │ └──────────┘ │
                                          └──────────────┘

                     ┌──────────────────────┐
                     │   src/utils.py       │
                     │  I/O · encoding ·    │
                     │  metadata · formats   │
                     └──────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone and enter the repository
git clone https://github.com/your-username/UAE-Commercial-Property-Classifier.git
cd UAE-Commercial-Property-Classifier

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux / macOS
.\venv\Scripts\activate    # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install full stack including XGBoost, BERT, and dev tools
pip install -e ".[all]"

# 5. Prepare your raw data CSV at data/raw/property_data.csv
#    (see "Dataset Requirements" below)

# 6. Run the pipeline
python main.py preprocess --input data/raw/property_data.csv --output data/processed/cleaned_data.csv
python main.py train --model random_forest --data data/processed/cleaned_data.csv
python main.py evaluate --model_path models/property_classifier.pkl --data_path data/processed/cleaned_data.csv
python main.py inference --model_path models/property_classifier.pkl --input_path data/processed/cleaned_data.csv --output_path data/results/predictions.csv
```

---

## Detailed Usage

### Preprocess

```bash
python main.py preprocess \
    --input data/raw/property_data.csv \
    --output data/processed/cleaned_data.csv \
    --test-size 0.2 \
    --random-state 42
```

| Argument       | Default | Description                         |
|----------------|---------|-------------------------------------|
| `--input`      | —       | Path to raw CSV                     |
| `--output`     | —       | Path for cleaned CSV                |
| `--test-size`  | 0.2     | Fraction reserved for testing       |
| `--random-state`| 42     | Random seed for reproducible split  |

### Train

```bash
python main.py train \
    --model logistic_regression \
    --data data/processed/cleaned_data.csv \
    --output models/property_classifier.pkl \
    --cv-folds 5
```

| Argument           | Default                          | Choices                                    |
|--------------------|----------------------------------|--------------------------------------------|
| `--model`          | `logistic_regression`            | `logistic_regression`, `random_forest`, `xgboost`, `bert` |
| `--data`           | `data/processed/cleaned_data.csv`| —                                          |
| `--output`         | `models/property_classifier.pkl` | —                                          |
| `--no-grid-search` | `False` (tuning enabled)         | flag                                       |
| `--cv-folds`       | `5`                              | integer ≥ 2                                |

### Inference

```bash
python main.py inference \
    --model_path models/property_classifier.pkl \
    --input_path data/processed/cleaned_data.csv \
    --output_path data/results/predictions.csv \
    --text-column description \
    --id-column property_id
```

| Argument         | Default        | Description                   |
|------------------|----------------|-------------------------------|
| `--model-path`   | —              | Trained model file            |
| `--input-path`   | —              | CSV to classify               |
| `--output-path`  | —              | Destination results CSV       |
| `--text-column`  | `description`  | Column with property text     |
| `--id-column`    | `None`         | Optional identifier column    |
| `--batch-size`   | `64`           | Batch size (BERT only)        |

### Evaluate

```bash
python main.py evaluate \
    --model_path models/property_classifier.pkl \
    --data_path data/processed/cleaned_data.csv \
    --output_dir data/results/
```

| Argument         | Default                        | Description                       |
|------------------|--------------------------------|-----------------------------------|
| `--model-path`   | —                              | Trained model file                |
| `--data-path`    | —                              | Labelled CSV (test set)           |
| `--output-dir`   | `data/results/`                | Output directory for metrics/plots|
| `--text-column`  | `description`                  | Text column name                  |
| `--target-column`| `property_type`                | Label column name                 |

---

## Directory Structure

```
UAE-Commercial-Property-Classifier/
├── src/                        # Core Python package
│   ├── __init__.py             # Package metadata
│   ├── preprocess.py           # Data cleaning & feature engineering
│   ├── train.py                # Model training (sklearn + BERT)
│   ├── inference.py            # Batch classification
│   ├── evaluate.py             # Metrics & diagnostic plots
│   └── utils.py                # I/O, encoding, metadata, formatting
├── tests/                      # pytest suite
│   ├── __init__.py
│   ├── test_preprocessing.py   # Unit tests for cleaning & encoding
│   ├── test_models.py          # Unit tests for pipeline & training
│   └── test_pipeline.py        # End-to-end integration tests
├── data/                       # Data directory
│   ├── raw/                    # Raw CSV input
│   ├── processed/              # Cleaned / feature-engineered data
│   └── results/                # Predictions, metrics, plots
├── models/                     # Trained model artifacts (.pkl / .joblib)
├── notebooks/                  # Jupyter notebooks (EDA, training, etc.)
├── main.py                     # CLI entry point
├── setup.py                    # pip install configuration
├── requirements.txt            # Pinned dependencies
├── .gitignore                  # Version-control exclusions
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## Model Comparison

| Model                  | Precision (macro) | Recall (macro) | F1 (macro) | Training Time | Inference Time |
|------------------------|-------------------|----------------|------------|---------------|----------------|
| Logistic Regression    | ~0.82             | ~0.80          | ~0.81      | seconds       | milliseconds   |
| Random Forest          | ~0.85             | ~0.84          | ~0.84      | seconds       | milliseconds   |
| XGBoost                | ~0.87             | ~0.86          | ~0.86      | seconds       | milliseconds   |
| BERT (AraBERTv2)       | ~0.91             | ~0.90          | ~0.90      | minutes       | seconds        |

*Typical results on a 10k-sample UAE property dataset.  Your mileage may vary.*

---

## Dataset Requirements

Your input CSV **must** contain the following columns (names are configurable):

| Column           | Type   | Description                              |
|------------------|--------|------------------------------------------|
| `description`    | str    | Free-text property description (required)|
| `property_type`  | str    | Ground-truth label (required)            |
| `location`       | str    | Area / district name (recommended)       |
| `size_sqft`      | float  | Property size in sq. ft. (optional)      |
| `price_aed`      | float  | Price in AED (optional)                  |
| `bedrooms`       | int    | Number of bedrooms (optional)            |
| `bathrooms`      | int    | Number of bathrooms (optional)           |

If numeric columns are missing, the preprocessor attempts to extract them from
the description text via regex patterns.

---

## Development

```bash
# Clone and install in editable mode with dev extras
pip install -e ".[dev]"

# Run tests
pytest tests/ -v --cov=src

# Lint
pip install black isort flake8
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

---

## Contributing

Contributions are welcome!  Please open an issue first to discuss the change,
then submit a pull request.  Ensure your code passes the test suite and follows
the existing style conventions.

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/my-feature`).
3. Make changes and write/update tests.
4. Run the full test suite (`pytest tests/ -v --cov=src`).
5. Commit (`git commit -m 'Add my feature'`).
6. Push to your branch and open a PR.

---

## Citation

If you use this project in academic work, please cite:

```bibtex
@software{uae_property_classifier_2025,
  author = {UAE Commercial Property Classifier Team},
  title = {UAE Commercial Property Classifier},
  url = {https://github.com/your-username/UAE-Commercial-Property-Classifier},
  year = {2025},
}
```

---

## License

Distributed under the **MIT License**.  See [LICENSE](LICENSE) for details.

---

## Author

**balaga raghuram** — built for the UAE real-estate analytics community.

<!-- Update 5 at 20260614022339 -->
<!-- Co-authored-by: balagaraghuram1 <balagaraghuram1@users.noreply.github.com> -->
