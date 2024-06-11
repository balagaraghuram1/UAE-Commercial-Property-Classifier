"""
Evaluation — compute classification metrics and render diagnostic plots.

Supports both binary and multi-class classification with accuracy,
precision, recall, F1-score, confusion matrix, and ROC-AUC (one-vs-rest).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.inference import _load_artifact
from src.utils import (
    format_metrics_table,
    load_dataframe,
    load_model,
    save_dataframe,
    save_metadata,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Plotting (optional — falls back gracefully when matplotlib not available)
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
except ImportError:
    _PLOTTING_AVAILABLE = False
    logger.warning("matplotlib / seaborn not installed — plots will not be generated.")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TEXT_COLUMN = "description"
DEFAULT_TARGET_COLUMN = "property_type"

MetricDict = Dict[str, float]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Calculate a comprehensive set of classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_proba: Predicted class probabilities (shape
            ``(n_samples, n_classes)``).  Required for ROC-AUC.
        class_names: Human-readable class names.

    Returns:
        Dictionary with keys:
            - ``accuracy``
            - ``precision`` (macro)
            - ``recall`` (macro)
            - ``f1`` (macro)
            - ``classification_report`` (string)
            - ``confusion_matrix`` (2-D list)
            - ``per_class_metrics`` (list of dicts, one per class)
            - ``roc_auc`` (macro, only when *y_proba* is provided)
    """
    metrics: Dict[str, Any] = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0, output_dict=False,
    )
    metrics["classification_report"] = report

    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    # Per-class metrics
    per_class = []
    labels = sorted(set(y_true) | set(y_pred))
    for cls in labels:
        per_class.append({
            "class": str(class_names[cls]) if class_names is not None else str(cls),
            "precision": float(precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]),
            "recall": float(recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]),
            "f1": float(f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]),
            "support": int((y_true == cls).sum()),
        })
    metrics["per_class_metrics"] = per_class

    # ROC-AUC (one-vs-rest)
    if y_proba is not None and y_proba.shape[1] > 2:
        try:
            metrics["roc_auc"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
            )
        except (ValueError, NotImplementedError) as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)
    elif y_proba is not None and y_proba.shape[1] == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
        except (ValueError, NotImplementedError) as exc:
            logger.warning("Could not compute ROC-AUC: %s", exc)

    return metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
    title: str = "Confusion Matrix",
) -> None:
    """Render a confusion matrix heatmap and save to disk.

    Args:
        cm: Confusion matrix array.
        class_names: Class labels.
        output_path: Destination image path (e.g. ``.png``).
        title: Plot title.
    """
    if not _PLOTTING_AVAILABLE:
        logger.warning("Cannot plot confusion matrix — matplotlib/seaborn not installed.")
        return

    plt.figure(figsize=(max(6, len(class_names) * 0.8), max(5, len(class_names) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(Path(output_path))
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    output_path: Union[str, Path],
) -> None:
    """Plot ROC curves (one-vs-rest) for each class.

    Args:
        y_true: Ground-truth integer labels.
        y_proba: Predicted probabilities, shape ``(n_samples, n_classes)``.
        class_names: Human-readable class names.
        output_path: Destination image path.
    """
    if not _PLOTTING_AVAILABLE:
        logger.warning("Cannot plot ROC curve — matplotlib/seaborn not installed.")
        return

    from sklearn.metrics import RocCurveDisplay

    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        y_true_bin = (y_true == i).astype(int)
        RocCurveDisplay.from_predictions(
            y_true_bin, y_proba[:, i], name=name, ax=ax,
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title("ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(Path(output_path))
    plt.close()
    logger.info("ROC curve saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------


def evaluate_classifier(
    model_path: Union[str, Path],
    data_path: Union[str, Path],
    output_dir: Union[str, Path] = "data/results/",
    text_column: str = DEFAULT_TEXT_COLUMN,
    target_column: str = DEFAULT_TARGET_COLUMN,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a trained classifier on a held-out test set.

    Args:
        model_path: Path to the trained model artifact.
        data_path: CSV with preprocessed data (must contain the target column
            and text column).
        output_dir: Directory where metrics JSON, plots, and result CSV will
            be saved.
        text_column: Column with property descriptions.
        target_column: Column with ground-truth labels.
        metrics: Deprecated — kept for backward compatibility; all available
            metrics are computed automatically.

    Returns:
        Dictionary of computed metrics.

    Raises:
        FileNotFoundError: If *model_path* or *data_path* does not exist.
        ValueError: If required columns are missing.
    """
    logger.info("Evaluation started — model: %s, data: %s", model_path, data_path)

    model_path = Path(model_path)
    data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, label_encoder, class_names, _ = _load_artifact(model_path)

    # Load data
    df = load_dataframe(data_path)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in {data_path}")
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' not found in {data_path}")

    y_true = df[target_column].values

    # Encode labels if needed
    if label_encoder is not None:
        y_true_enc = label_encoder.transform(y_true)
        classes = label_encoder.classes_
    else:
        y_true_enc = y_true
        classes = class_names or sorted(set(y_true))

    texts = df[text_column].fillna("").tolist()

    # Predict
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(texts)
    else:
        y_proba = None
    y_pred = model.predict(texts)

    # Compute metrics
    results = compute_classification_metrics(y_true_enc, y_pred, y_proba, classes)

    # Log summary
    summary = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1": results["f1"],
    }
    if "roc_auc" in results:
        summary["roc_auc"] = results["roc_auc"]

    logger.info("\n%s", format_metrics_table(summary, title="Evaluation Results"))

    # Persist metrics
    metrics_path = output_dir / "evaluation_metrics.json"
    save_metadata(results, metrics_path)

    # Save predictions with probabilities
    pred_df = df.copy()
    pred_df["predicted_label"] = y_pred
    if y_proba is not None and classes is not None:
        for i, cls in enumerate(classes):
            pred_df[f"prob_{cls}"] = y_proba[:, i]
    pred_path = output_dir / "evaluation_predictions.csv"
    save_dataframe(pred_df, pred_path)

    # Plots
    if _PLOTTING_AVAILABLE:
        cm = np.array(results["confusion_matrix"])
        plot_confusion_matrix(
            cm, list(classes),
            output_path=output_dir / "confusion_matrix.png",
        )
        if y_proba is not None and len(classes) > 1:
            plot_roc_curve(
                y_true_enc, y_proba, list(classes),
                output_path=output_dir / "roc_curve.png",
            )

    logger.info("Evaluation complete — results saved to %s", output_dir.resolve())
    return results
