"""
Model training — Logistic Regression, Random Forest, XGBoost, and BERT.

Each model is trained with cross-validation and optional hyperparameter
tuning via ``GridSearchCV``.  Persisted artifacts include the trained
estimator, label encoder, vectoriser (if any), and training metadata.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.utils import (
    build_metadata,
    fit_label_encoder,
    load_dataframe,
    save_dataframe,
    save_metadata,
    save_model,
)

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# BERT imports (optional — graceful fallback when transformers/torch missing)
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False
    logger.warning("transformers / torch not installed — BERT model unavailable.")

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("xgboost not installed — XGBoost model unavailable.")


# ---------------------------------------------------------------------------
# Supported models registry
# ---------------------------------------------------------------------------

SUPPORTED_MODELS = ["logistic_regression", "random_forest", "xgboost", "bert"]
TEXT_FEATURE_COLUMN = "description"

# Default hyperparameter grids for GridSearchCV
DEFAULT_PARAM_GRIDS: Dict[str, Dict[str, List[Any]]] = {
    "logistic_regression": {
        "clf__C": [0.01, 0.1, 1.0, 10.0],
        "clf__solver": ["liblinear", "lbfgs"],
        "clf__max_iter": [100, 300],
    },
    "random_forest": {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": [2, 5],
    },
    "xgboost": {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 6, 9],
        "clf__learning_rate": [0.01, 0.1, 0.3],
        "clf__subsample": [0.8, 1.0],
    },
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingResult:
    """Outcome of a training run."""

    model_type: str
    model: Any
    label_encoder: Any
    vectorizer: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    cv_scores: Optional[List[float]] = None
    best_params: Optional[Dict[str, Any]] = None
    label_classes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_pipeline(
    model_type: str,
    n_classes: int,
) -> Pipeline:
    """Construct a sklearn ``Pipeline`` with a TF-IDF vectoriser + classifier.

    Args:
        model_type: One of ``SUPPORTED_MODELS``.
        n_classes: Number of classes (affects some model configs).

    Returns:
        A composable ``Pipeline``.
    """
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)

    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=42)
    elif model_type == "random_forest":
        clf = RandomForestClassifier(random_state=42)
    elif model_type == "xgboost":
        if not _XGB_AVAILABLE:
            raise ImportError("XGBoost is required but not installed.")
        clf = xgb.XGBClassifier(
            objective="multi:softprob",
            num_class=n_classes,
            random_state=42,
            verbosity=0,
        )
    else:
        raise ValueError(
            f"Unknown scikit-learn-compatible model: {model_type}. "
            f"Supported: {SUPPORTED_MODELS}"
        )

    return Pipeline([("tfidf", tfidf), ("clf", clf)])


# ---------------------------------------------------------------------------
# BERT trainer (internal)
# ---------------------------------------------------------------------------


def _train_bert(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    label_encoder: Any,
    output_dir: Union[str, Path],
    model_name: str = "aubmindlab/bert-base-arabertv2",
    **kwargs: Any,
) -> TrainingResult:
    """Train a BERT-based sequence classifier on UAE property descriptions.

    Args:
        X_train: Training features (must contain a ``description`` column).
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        label_encoder: Sklearn ``LabelEncoder`` fitted on labels.
        output_dir: Where to save the model and tokenizer.
        model_name: HuggingFace model identifier.  Defaults to AraBERTv2.
        **kwargs: Extra ``TrainingArguments`` overrides.

    Returns:
        A ``TrainingResult``.
    """
    if not _BERT_AVAILABLE:
        raise ImportError("BERT training requires `transformers` and `torch`.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_labels = len(label_encoder.classes_)
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)

    train_texts = X_train[TEXT_FEATURE_COLUMN].fillna("").tolist()
    val_texts = X_val[TEXT_FEATURE_COLUMN].fillna("").tolist()

    train_enc = tokenizer(train_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
    val_enc = tokenizer(val_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    class PropertyDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
            return item

        def __len__(self):
            return len(self.labels)

    train_labels = label_encoder.transform(y_train).tolist()
    val_labels = label_encoder.transform(y_val).tolist()

    train_dataset = PropertyDataset(train_enc, train_labels)
    val_dataset = PropertyDataset(val_enc, val_labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        **kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    trainer.save_model(str(output_dir / "bert_final"))
    tokenizer.save_pretrained(str(output_dir / "bert_final"))

    result = TrainingResult(
        model_type="bert",
        model=model,
        label_encoder=label_encoder,
        vectorizer=None,
        metadata={"model_name": model_name, "output_dir": str(output_dir)},
        label_classes=label_encoder.classes_.tolist(),
    )
    return result


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------


def train_property_classifier(
    model_type: str = "logistic_regression",
    data_path: Union[str, Path] = "data/processed/cleaned_data.csv",
    model_output_path: Union[str, Path] = "models/property_classifier.pkl",
    metadata_output_path: Optional[Union[str, Path]] = None,
    text_column: str = "description",
    target_column: str = "property_type",
    use_grid_search: bool = True,
    cv_folds: int = 5,
    param_grid: Optional[Dict[str, List[Any]]] = None,
    **kwargs: Any,
) -> TrainingResult:
    """Train a property classifier and persist the artifact.

    Args:
        model_type: Model family (``logistic_regression``, ``random_forest``,
            ``xgboost``, or ``bert``).
        data_path: CSV with preprocessed data (includes both train and test
            partitions if a ``split`` column is present; otherwise a random
            split is performed inside this function).
        model_output_path: Destination for the persisted model.
        metadata_output_path: Optional separate JSON file for metadata.
        text_column: Name of the text description column.
        target_column: Name of the label column.
        use_grid_search: Whether to run ``GridSearchCV`` for hyperparameter
            tuning.
        cv_folds: Number of cross-validation folds.
        param_grid: Custom hyperparameter grid (falls back to
            ``DEFAULT_PARAM_GRIDS``).
        **kwargs: Additional arguments passed to the underlying estimator or
            BERT ``TrainingArguments``.

    Returns:
        A ``TrainingResult`` dataclass instance.

    Raises:
        FileNotFoundError: If *data_path* does not exist.
        ValueError: If *model_type* is unsupported.
    """
    logger.info("Training started — model: %s, data: %s", model_type, data_path)

    if model_type not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model: {model_type}. Choose from {SUPPORTED_MODELS}"
        )

    # Load data
    df = load_dataframe(data_path)
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in {data_path}")
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {data_path}")

    # Handle pre-existing split (keyed by "split" in the CSV)
    if "split" in df.columns:
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[target_column],
        )

    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]
    X_val = test_df.drop(columns=[target_column])
    y_val = test_df[target_column]

    # Label encoding
    le = fit_label_encoder(y_train)
    y_train_enc = le.transform(y_train)
    n_classes = len(le.classes_)

    metadata = build_metadata(
        model_type=model_type,
        n_classes=n_classes,
        classes=le.classes_.tolist(),
        train_samples=len(X_train),
        val_samples=len(X_val),
        use_grid_search=use_grid_search,
    )

    # BERT path
    if model_type == "bert":
        result = _train_bert(
            X_train, y_train, X_val, y_val, le,
            output_dir=Path(model_output_path).parent,
            **kwargs,
        )
        save_model(result.model, model_output_path, metadata=metadata)
        if metadata_output_path:
            save_metadata(metadata, metadata_output_path)
        logger.info("BERT training complete — model saved to %s", model_output_path)
        return result

    # sklearn pipeline path
    pipeline = _build_pipeline(model_type, n_classes)
    X_train_text = X_train[text_column].fillna("").tolist()
    X_val_text = X_val[text_column].fillna("").tolist()

    # Cross-validation
    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(
        pipeline, X_train_text, y_train_enc,
        cv=cv_strategy, scoring="f1_macro",
    )
    metadata["cv_f1_macro_mean"] = float(cv_scores.mean())
    metadata["cv_f1_macro_std"] = float(cv_scores.std())
    logger.info("CV F1 (macro): %0.4f (±%0.4f)", cv_scores.mean(), cv_scores.std())

    # GridSearchCV
    best_params = None
    if use_grid_search:
        grid = param_grid or DEFAULT_PARAM_GRIDS.get(model_type)
        if grid:
            gs = GridSearchCV(
                pipeline,
                param_grid=grid,
                cv=cv_folds,
                scoring="f1_macro",
                n_jobs=-1,
                verbose=0,
            )
            gs.fit(X_train_text, y_train_enc)
            pipeline = gs.best_estimator_
            best_params = gs.best_params_
            metadata["best_params"] = best_params
            metadata["best_cv_score"] = float(gs.best_score_)
            logger.info("GridSearch best params: %s (score: %0.4f)", best_params, gs.best_score_)
        else:
            pipeline.fit(X_train_text, y_train_enc)
    else:
        pipeline.fit(X_train_text, y_train_enc)

    # Persist
    artifact = {
        "model": pipeline,
        "label_encoder": le,
    }
    save_model(artifact, model_output_path, metadata=metadata)

    if metadata_output_path:
        save_metadata(metadata, metadata_output_path)

    result = TrainingResult(
        model_type=model_type,
        model=pipeline,
        label_encoder=le,
        vectorizer=pipeline.named_steps.get("tfidf"),
        metadata=metadata,
        cv_scores=cv_scores.tolist(),
        best_params=best_params,
        label_classes=le.classes_.tolist(),
    )

    logger.info("Training complete — model saved to %s", model_output_path)
    return result
