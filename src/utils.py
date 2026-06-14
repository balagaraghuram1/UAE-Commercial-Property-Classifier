"""
utilities for data loading, encoding, metadata tracking, and result formatting.

Centralises reusable helpers shared across the preprocessing, training,
inference, and evaluation modules.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_dataframe(
    path: Union[str, Path],
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """Load a CSV dataset from *path* with sensible defaults.

    Args:
        path: File path (string or ``Path``).
        encoding: File encoding (default ``utf-8``).
        **kwargs: Additional arguments forwarded to ``pd.read_csv``.

    Returns:
        Loaded DataFrame.

    Raises:
        FileNotFoundError: When the file does not exist.
        pd.errors.EmptyDataError: When the file is empty.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path.resolve()}")
    logger.info("Loading data from %s", path.resolve())
    return pd.read_csv(path, encoding=encoding, **kwargs)


def save_dataframe(
    df: pd.DataFrame,
    path: Union[str, Path],
    index: bool = False,
    **kwargs: Any,
) -> None:
    """Write a DataFrame to CSV.

    Automatically creates parent directories if they do not exist.

    Args:
        df: DataFrame to persist.
        path: Destination file path.
        index: Whether to write the row index (default ``False``).
        **kwargs: Additional arguments forwarded to ``df.to_csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index, **kwargs)
    logger.info("Saved DataFrame (%d rows) to %s", len(df), path.resolve())


def save_model(
    model: Any,
    path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a trained model (and optional metadata) with ``joblib``.

    Args:
        model: Trained estimator.
        path: Destination path (``.pkl`` or ``.joblib`` extension).
        metadata: Dictionary of extra information stored alongside the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact: Dict[str, Any] = {"model": model}
    if metadata:
        artifact["metadata"] = metadata
    joblib.dump(artifact, path)
    logger.info("Model saved to %s", path.resolve())


def load_model(
    path: Union[str, Path],
) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Load a previously persisted model artifact.

    Args:
        path: Path to the ``.pkl`` / ``.joblib`` file.

    Returns:
        A tuple ``(model, metadata)`` where *metadata* is ``None`` when no
        metadata was stored.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.resolve()}")
    artifact: Dict[str, Any] = joblib.load(path)
    model = artifact["model"]
    metadata = artifact.get("metadata")
    logger.info("Model loaded from %s", path.resolve())
    return model, metadata


# ---------------------------------------------------------------------------
# Label encoding helpers
# ---------------------------------------------------------------------------

def fit_label_encoder(
    series: pd.Series,
    le: Optional[LabelEncoder] = None,
) -> LabelEncoder:
    """Fit a ``LabelEncoder`` on *series*.

    Args:
        series: Target labels.
        le: Optional pre-existing encoder (fitted in-place).

    Returns:
        Fitted ``LabelEncoder``.
    """
    le = le or LabelEncoder()
    le.fit(series)
    logger.info("Label encoding fitted — %d classes", len(le.classes_))
    return le


def encode_labels(
    series: pd.Series,
    le: LabelEncoder,
) -> np.ndarray:
    """Transform *series* with a fitted ``LabelEncoder``.

    Args:
        series: Labels to encode.
        le: Fitted encoder.

    Returns:
        Numpy array of encoded integer labels.
    """
    return le.transform(series)


def decode_labels(
    encoded: Sequence[int],
    le: LabelEncoder,
) -> List[str]:
    """Reverse label encoding back to original strings.

    Args:
        encoded: Integer-encoded labels.
        le: Fitted encoder.

    Returns:
        List of original label strings.
    """
    return le.inverse_transform(encoded).tolist()


# ---------------------------------------------------------------------------
# Metadata tracking
# ---------------------------------------------------------------------------

def build_metadata(
    model_type: str,
    **extra: Any,
) -> Dict[str, Any]:
    """Construct a standard metadata dictionary for model tracking.

    Args:
        model_type: Short name of the model (e.g. ``random_forest``).
        **extra: Arbitrary extra key-value pairs.

    Returns:
        Metadata dictionary.
    """
    return {
        "model_type": model_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "package_version": __import__("src").__version__,
        **extra,
    }


def save_metadata(
    metadata: Dict[str, Any],
    path: Union[str, Path],
) -> None:
    """Write metadata as JSON.

    Args:
        metadata: Dictionary to persist.
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)
    logger.info("Metadata saved to %s", path.resolve())


def load_metadata(path: Union[str, Path]) -> Dict[str, Any]:
    """Load metadata from a JSON file.

    Args:
        path: Path to the JSON metadata file.

    Returns:
        Metadata dictionary.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_metrics_table(metrics: Dict[str, float], title: str = "Metrics") -> str:
    """Pretty-print a metrics dictionary as a centered table.

    Args:
        metrics: Mapping of metric name → value.
        title: Optional table title.

    Returns:
        Formatted string table.
    """
    if not metrics:
        return "(no metrics)"

    name_width = max(len(k) for k in metrics)
    val_width = max(len(f"{v:.4f}") for v in metrics.values())
    sep = "+" + "-" * (name_width + 2) + "+" + "-" * (val_width + 2) + "+"
    header = f"| {'Metric'.ljust(name_width)} | {'Value'.rjust(val_width)} |"
    rows = "\n".join(
        f"| {k.ljust(name_width)} | {f'{v:.4f}'.rjust(val_width)} |"
        for k, v in metrics.items()
    )
    title_line = f"  {title}  ".center(len(sep) - 2)
    return f"\n{title_line}\n{sep}\n{header}\n{sep}\n{rows}\n{sep}"


def results_to_dataframe(
    predictions: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    **extra_columns: Any,
) -> pd.DataFrame:
    """Build a results DataFrame from prediction arrays.

    Args:
        predictions: Predicted class indices or labels.
        probabilities: Confidence scores, shape ``(n_samples, n_classes)``.
        class_names: Optional class names for human-readable output.
        **extra_columns: Additional columns to include (e.g. identifiers).

    Returns:
        DataFrame with columns for predictions, confidence, and extras.
    """
    data: Dict[str, Any] = {}
    data.update(extra_columns)

    if probabilities is not None and class_names:
        for i, name in enumerate(class_names):
            data[f"prob_{name}"] = probabilities[:, i]
        data["confidence"] = probabilities.max(axis=1)
    elif probabilities is not None:
        data["confidence"] = probabilities.max(axis=1)

    data["prediction"] = predictions
    if class_names:
        data["prediction_label"] = [
            class_names[int(p)] for p in predictions
        ]

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: Union[str, Path]) -> Path:
    """Create *path* as a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Resolved ``Path`` object.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


def resolve_data_path(relative: str) -> Path:
    """Resolve a path relative to the project root (two levels up from ``src``).

    Args:
        relative: Relative path (e.g. ``data/raw/properties.csv``).

    Returns:
        Absolute ``Path``.
    """
    root = Path(__file__).resolve().parent.parent
    return root / relative
