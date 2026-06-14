"""Integration tests for the full training → inference → evaluation pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.evaluate import compute_classification_metrics
from src.inference import classify_properties
from src.preprocess import preprocess_property_data
from src.train import train_property_classifier


@pytest.fixture(scope="module")
def synthetic_dataset(tmp_path_factory) -> str:
    """Generate a small synthetic dataset and run preprocessing."""
    tmp = tmp_path_factory.mktemp("pipeline")
    n = 100
    np.random.seed(42)
    types = ["Villa", "Apartment", "Office"]
    df = pd.DataFrame({
        "description": [
            f"This is a {' '.join(np.random.choice(['luxury', 'spacious', 'modern', 'affordable', 'premium'], 3).tolist())} "
            f"commercial property located in {np.random.choice(['Dubai Marina', 'Downtown Dubai', 'Abu Dhabi Corniche'])} "
            f"with {np.random.randint(1, 6)} bedrooms and {np.random.randint(1000, 10000)} sqft"
            for _ in range(n)
        ],
        "location": np.random.choice(
            ["Dubai Marina", "Downtown Dubai", "Abu Dhabi Corniche", "Sharjah"], n,
        ),
        "property_type": np.random.choice(types, n),
        "size_sqft": np.random.randint(800, 15000, n).astype(float),
        "price_aed": np.random.randint(500_000, 10_000_000, n).astype(float),
        "bedrooms": np.random.randint(1, 6, n),
        "bathrooms": np.random.randint(1, 5, n),
    })
    raw_path = tmp / "raw.csv"
    df.to_csv(raw_path, index=False)
    return str(raw_path)


class TestEndToEndPipeline:
    """Complete pipeline test: preprocess → train → inference → evaluate."""

    def test_full_pipeline(self, synthetic_dataset, tmp_path) -> None:
        # 1. Preprocess
        clean_path = tmp_path / "cleaned.csv"
        model_path = tmp_path / "model.pkl"
        pred_path = tmp_path / "predictions.csv"
        eval_dir = tmp_path / "eval"

        X_train, X_test, y_train, y_test = preprocess_property_data(
            synthetic_dataset, clean_path, test_size=0.3, random_state=42,
        )
        assert len(X_train) > 0
        assert len(X_test) > 0

        # 2. Train
        result = train_property_classifier(
            model_type="logistic_regression",
            data_path=str(clean_path),
            model_output_path=str(model_path),
            use_grid_search=False,
            cv_folds=2,
        )
        assert result.model is not None
        assert model_path.exists()

        # 3. Inference
        pred_df = classify_properties(
            model_path=str(model_path),
            input_path=str(clean_path),
            output_path=str(pred_path),
        )
        assert len(pred_df) > 0
        assert "prediction" in pred_df.columns
        assert pred_path.exists()

        # 4. Evaluate
        from src.evaluate import evaluate_classifier
        metrics = evaluate_classifier(
            model_path=str(model_path),
            data_path=str(clean_path),
            output_dir=str(eval_dir),
        )
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.0
        assert metrics["accuracy"] <= 1.0
        assert "classification_report" in metrics
        assert "per_class_metrics" in metrics


class TestComputeMetrics:
    def test_binary_metrics(self) -> None:
        y_true = np.array([0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 1])
        y_proba = np.array([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4], [0.9, 0.1], [0.2, 0.8]])
        metrics = compute_classification_metrics(y_true, y_pred, y_proba, ["A", "B"])
        assert metrics["accuracy"] == 0.8
        assert "roc_auc" in metrics

    def test_multi_class_metrics(self) -> None:
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.eye(6)[y_true] * 0.9 + 0.1 / 3
        metrics = compute_classification_metrics(y_true, y_pred, y_proba, ["X", "Y", "Z"])
        assert metrics["accuracy"] == 1.0
        assert len(metrics["per_class_metrics"]) == 3

    def test_per_class_metrics_include_support(self) -> None:
        y_true = np.array([0, 0, 1, 1, 2])
        y_pred = np.array([0, 0, 1, 1, 2])
        metrics = compute_classification_metrics(y_true, y_pred, None, ["A", "B", "C"])
        per = {m["class"]: m["support"] for m in metrics["per_class_metrics"]}
        assert per["A"] == 2
        assert per["B"] == 2
        assert per["C"] == 1
