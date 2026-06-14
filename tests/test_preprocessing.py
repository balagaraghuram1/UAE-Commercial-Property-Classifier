"""Unit tests for the preprocessing module."""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocess import (
    clean_text,
    clean_text_series,
    encode_categorical,
    engineer_features,
    extract_price,
    extract_rooms,
    extract_size_sqft,
    handle_missing_values,
    preprocess_property_data,
)


class TestCleanText:
    def test_lowercase(self) -> None:
        assert clean_text("Luxury Villa") == "luxury villa"

    def test_whitespace_collapse(self) -> None:
        assert clean_text("big    space") == "big space"

    def test_arabic_removed(self) -> None:
        result = clean_text("villa فيلا for rent")
        assert "rent" in result
        assert "فيلا" not in result

    def test_special_chars_removed(self) -> None:
        assert clean_text("property!!! @ 100%") == "property 100"

    def test_nan_returns_empty(self) -> None:
        assert clean_text(None) == ""
        assert clean_text(float("nan")) == ""

    def test_series_cleaning(self) -> None:
        s = pd.Series(["HELLO", None, "MIXED نص"])
        result = clean_text_series(s)
        assert result.iloc[0] == "hello"
        assert result.iloc[1] == ""
        assert "mixed" in result.iloc[2]
        assert "نص" not in result.iloc[2]


class TestNumericExtraction:
    def test_size_sqft(self) -> None:
        s = pd.Series(["2500 sqft", "100 sq. ft.", "1500 square feet", "no size"])
        result = extract_size_sqft(s)
        assert result.iloc[0] == 2500.0
        assert result.iloc[1] == 100.0
        assert result.iloc[2] == 1500.0
        assert pd.isna(result.iloc[3])

    def test_price(self) -> None:
        s = pd.Series(["1,200,000 AED", "500000", "no price"])
        result = extract_price(s)
        assert result.iloc[0] == 1200000.0
        assert result.iloc[1] == 500000.0
        assert pd.isna(result.iloc[2])

    def test_rooms(self) -> None:
        s = pd.Series(["3 bedroom", "2 bed", "5 rooms", "studio"])
        result = extract_rooms(s)
        assert result.iloc[0] == 3
        assert result.iloc[1] == 2
        assert result.iloc[2] == 5
        assert pd.isna(result.iloc[3])


class TestCategoricalEncoding:
    def test_fit_and_transform(self) -> None:
        df = pd.DataFrame({"location": ["Dubai", "Abu Dhabi", "Dubai"]})
        encoded, mappings = encode_categorical(df, ["location"])
        assert encoded["location"].tolist() == [0, 1, 0]
        assert "location" in mappings

    def test_with_precomputed_mappings(self) -> None:
        df = pd.DataFrame({"location": ["Dubai", "Sharjah"]})
        pre = {"location": {"Dubai": 0, "Abu Dhabi": 1, "UNKNOWN": -1}}
        encoded, _ = encode_categorical(df, ["location"], encodings=pre)
        assert encoded["location"].tolist() == [0, -1]


class TestMissingValues:
    def test_numeric_median(self) -> None:
        df = pd.DataFrame({"price": [100.0, None, 300.0], "label": ["a", None, "b"]})
        result = handle_missing_values(df, numeric_strategy="median")
        assert result["price"].iloc[1] == 200.0
        assert result["label"].iloc[1] == "UNKNOWN"


class TestFeatureEngineering:
    def test_price_per_sqft(self) -> None:
        df = pd.DataFrame({"price_aed": [1_000_000], "size_sqft": [2000]})
        result = engineer_features(df)
        assert result["price_per_sqft"].iloc[0] == 500.0

    def test_log_price(self) -> None:
        df = pd.DataFrame({"price_aed": [1_000_000]})
        result = engineer_features(df)
        assert result["log_price"].iloc[0] > 0


class TestEndToEnd:
    def test_preprocess_pipeline(self, tmp_path) -> None:
        data = {
            "description": [
                "Luxury 3 bedroom villa 2500 sqft in Dubai Marina",
                "Modern office 1500 sqft in Downtown Dubai",
                "Retail space 800 sqft in Abu Dhabi",
                "Another villa in Palm Jumeirah",
                "Office space in Business Bay",
                "Warehouse in Al Quoz",
                "Small shop in Deira",
                "Restaurant space in JLT",
            ],
            "location": [
                "Dubai Marina", "Downtown Dubai", "Abu Dhabi",
                "Palm Jumeirah", "Business Bay", "Al Quoz",
                "Deira", "JLT",
            ],
            "property_type": ["Villa", "Office", "Retail", "Villa", "Office", "Warehouse", "Retail", "Restaurant"],
            "size_sqft": [2500, 1500, 800, 5000, 2000, 10000, 400, 2000],
            "price_aed": [2_500_000, 1_200_000, 600_000, 8_000_000, 3_000_000, 5_000_000, 300_000, 1_500_000],
            "bedrooms": [3, 0, 0, 5, 0, 0, 0, 0],
            "bathrooms": [3, 1, 1, 4, 2, 2, 1, 2],
        }
        raw = tmp_path / "raw.csv"
        out = tmp_path / "clean.csv"
        pd.DataFrame(data).to_csv(raw, index=False)

        X_train, X_test, y_train, y_test = preprocess_property_data(
            raw, out, test_size=0.25, random_state=42,
        )

        assert len(X_train) >= 1
        assert len(X_test) >= 1
        assert out.exists()

    def test_preprocess_missing_columns(self, tmp_path) -> None:
        data = {
            "description": [
                "some text", "another property", "third listing",
                "fourth listing", "fifth property", "sixth item",
                "seventh listing", "eighth property",
            ],
            "property_type": ["Office", "Villa", "Office", "Villa", "Apartment", "Apartment", "Office", "Villa"],
        }
        raw = tmp_path / "minimal.csv"
        out = tmp_path / "clean_min.csv"
        pd.DataFrame(data).to_csv(raw, index=False)

        # Should not crash when expected columns are missing
        result = preprocess_property_data(raw, out, test_size=0.25, random_state=42)
        assert result is not None
