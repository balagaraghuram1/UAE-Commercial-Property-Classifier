"""
Inference — classify new commercial property descriptions using a trained model.

Supports both sklearn pipelines (with embedded TF-IDF vectoriser) and
standalone BERT models from the ``transformers`` library.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils import load_dataframe, load_model, results_to_dataframe, save_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BERT support (optional)
# ---------------------------------------------------------------------------

try:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    _BERT_AVAILABLE = True
except ImportError:
    _BERT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COLUMN = "description"


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _load_artifact(
    model_path: Union[str, Path],
) -> Tuple[Any, Optional[Any], Optional[List[str]], Optional[Dict[str, Any]]]:
    """Load a persisted model artifact, handling both sklearn and BERT.

    Args:
        model_path: Path to ``.pkl`` / ``.joblib`` file or a BERT directory.

    Returns:
        ``(model, label_encoder, class_names, metadata)``.

    Raises:
        FileNotFoundError: When *model_path* does not exist.
    """
    model_path = Path(model_path)

    # Try loading as a joblib artifact
    try:
        model, metadata = load_model(model_path)
    except Exception:
        model, metadata = None, None

    if model is not None:
        # sklearn pipeline saved as dict: {"model": pipeline, "label_encoder": le}
        if isinstance(model, dict):
            label_encoder = model.get("label_encoder")
            clf = model.get("model")
            class_names = label_encoder.classes_.tolist() if label_encoder is not None else None
            return clf, label_encoder, class_names, metadata
        # Direct sklearn pipeline
        return model, None, None, metadata

    # BERT model directory
    if _BERT_AVAILABLE and model_path.is_dir():
        logger.info("Loading BERT model from %s", model_path)
        config = AutoConfig.from_pretrained(str(model_path))
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        bert_model = AutoModelForSequenceClassification.from_pretrained(str(model_path), config=config)
        class_names = list(config.id2label.values()) if hasattr(config, "id2label") else None
        return (bert_model, tokenizer), None, class_names, None

    raise FileNotFoundError(f"Could not load model from {model_path}")


def _predict_sklearn(
    model: Any,
    texts: List[str],
    label_encoder: Optional[Any] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run prediction with an sklearn-compatible pipeline.

    Args:
        model: Fitted sklearn estimator or pipeline.
        texts: List of raw text descriptions.
        label_encoder: Fitted ``LabelEncoder`` for decoding.

    Returns:
        ``(pred_indices, probabilities, pred_labels)``.
    """
    # Pipeline handles its own vectorisation
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(texts)
    else:
        proba = None
    preds = model.predict(texts)

    classes = label_encoder.classes_ if label_encoder is not None else None
    if classes is not None:
        labels = [classes[int(p)] for p in preds]
    else:
        labels = [str(p) for p in preds]

    return np.asarray(preds), proba, labels


def _predict_bert(
    bert_artifacts: Tuple[Any, Any],
    texts: List[str],
    class_names: Optional[List[str]] = None,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Run prediction with a BERT model.

    Args:
        bert_artifacts: ``(model, tokenizer)`` tuple.
        texts: List of raw text descriptions.
        class_names: Human-readable class names.
        device: Device string (``'cpu'`` or ``'cuda'``).

    Returns:
        ``(pred_indices, probabilities, pred_labels)``.
    """
    bert_model, tokenizer = bert_artifacts
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    bert_model.to(device)
    bert_model.eval()

    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=128, return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**encodings)
        logits = outputs.logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()

    pred_indices = np.argmax(proba, axis=1)

    if class_names:
        labels = [class_names[int(i)] for i in pred_indices]
    else:
        labels = [str(i) for i in pred_indices]

    return pred_indices, proba, labels


# ---------------------------------------------------------------------------
# Main API
# ---------------------------------------------------------------------------


def classify_properties(
    model_path: Union[str, Path],
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    text_column: str = DEFAULT_TEXT_COLUMN,
    id_column: Optional[str] = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Run batch classification on a CSV of property descriptions.

    Args:
        model_path: Path to a trained model (``.pkl`` / ``.joblib`` file or
            BERT directory).
        input_path: CSV file containing property data to classify.
        output_path: Destination CSV file for results.
        text_column: Column name that holds the text description.
        id_column: Optional column name for record identifiers (e.g.
            ``property_id``).
        batch_size: Number of records per batch (BERT only).

    Returns:
        DataFrame with predictions, confidence scores, and probabilities.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If *text_column* is missing from the input CSV.
    """
    logger.info(
        "Inference started — model: %s, input: %s, output: %s",
        model_path, input_path, output_path,
    )

    df = load_dataframe(input_path)
    if text_column not in df.columns:
        raise ValueError(
            f"Text column '{text_column}' not found in input data. "
            f"Available columns: {list(df.columns)}"
        )

    model, label_encoder, class_names, metadata = _load_artifact(model_path)

    texts = df[text_column].fillna("").tolist()
    ids = df[id_column].tolist() if id_column and id_column in df.columns else None

    # Dispatch to correct predictor
    if isinstance(model, tuple) and _BERT_AVAILABLE:
        # BERT model
        all_preds: List[int] = []
        all_proba: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            preds, proba, _ = _predict_bert(model, batch, class_names)
            all_preds.extend(preds.tolist())
            all_proba.append(proba)
        final_preds = np.array(all_preds)
        final_proba = np.vstack(all_proba) if all_proba else None
    else:
        final_preds, final_proba, _ = _predict_sklearn(model, texts, label_encoder)

    # Build results DataFrame
    extra: Dict[str, Any] = {}
    if ids is not None:
        extra[id_column] = ids  # type: ignore[assignment]
    results_df = results_to_dataframe(final_preds, final_proba, class_names, **extra)

    save_dataframe(results_df, output_path)
    logger.info("Inference complete — %d records classified", len(results_df))
    return results_df
