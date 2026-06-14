"""Unit tests for the training module — pipeline construction and model training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.train import (
    SUPPORTED_MODELS,
    _build_pipeline,
    train_property_classifier,
)


class TestPipelineBuilding:
    def test_logistic_regression_pipeline(self) -> None:
        pipe = _build_pipeline("logistic_regression", n_classes=3)
        assert isinstance(pipe, Pipeline)
        assert "tfidf" in pipe.named_steps
        assert "clf" in pipe.named_steps

    def test_random_forest_pipeline(self) -> None:
        pipe = _build_pipeline("random_forest", n_classes=3)
        assert isinstance(pipe, Pipeline)

    def test_xgboost_pipeline_import_error(self) -> None:
        try:
            import xgboost  # noqa: F401
        except ImportError:
            pytest.skip("xgboost not installed — skipping")
        pipe = _build_pipeline("xgboost", n_classes=3)
        assert isinstance(pipe, Pipeline)

    def test_unsupported_model(self) -> None:
        with pytest.raises(ValueError):
            _build_pipeline("unknown_model", n_classes=3)


class TestTraining:
    @pytest.fixture
    def sample_dataframe(self, tmp_path) -> str:
        n = 50
        np.random.seed(42)
        df = pd.DataFrame({
            "description": [
                f"Sample property description {i} with details about space and location"
                for i in range(n)
            ],
            "property_type": np.random.choice(["Villa", "Apartment", "Office"], n),
        })
        path = tmp_path / "test_data.csv"
        df.to_csv(path, index=False)
        return str(path)

    def test_train_logistic_regression(self, sample_dataframe) -> None:
        result = train_property_classifier(
            model_type="logistic_regression",
            data_path=sample_dataframe,
            model_output_path=sample_dataframe.replace(".csv", "_model.pkl"),
            use_grid_search=False,
            cv_folds=2,
        )
        assert result.model is not None
        assert result.metadata is not None
        assert "model_type" in result.metadata

    def test_train_random_forest(self, sample_dataframe) -> None:
        result = train_property_classifier(
            model_type="random_forest",
            data_path=sample_dataframe,
            model_output_path=sample_dataframe.replace(".csv", "_rf_model.pkl"),
            use_grid_search=False,
            cv_folds=2,
        )
        assert result.model is not None
        assert len(result.label_classes) > 0

    def test_model_persistence(self, sample_dataframe) -> None:
        import joblib
        model_path = sample_dataframe.replace(".csv", "_persist.pkl")
        result = train_property_classifier(
            model_type="logistic_regression",
            data_path=sample_dataframe,
            model_output_path=model_path,
            use_grid_search=False,
            cv_folds=2,
        )
        artifact = joblib.load(model_path)
        assert "model" in artifact
        assert "metadata" in artifact
