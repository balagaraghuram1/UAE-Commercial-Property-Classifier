"""
Property data preprocessing — cleaning, feature engineering, encoding, and splitting.

Handles the idiosyncrasies of UAE real-estate listings including mixed
Arabic/English text, varied measurement units, and sparse categorical fields.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import load_dataframe, save_dataframe

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
NON_ALPHANUM_PATTERN = re.compile(r"[^a-zA-Z0-9\s_/\-]")
WHITESPACE_PATTERN = re.compile(r"\s+")
SIZE_PATTERN = re.compile(
    r"(?P<value>\d+[.,]?\d*)\s*(sq[\s.]?\s*(ft|foot|m|meter|meter|feet)|"
    r"square\s*(feet|meters|meter|foot))",
    re.IGNORECASE,
)
PRICE_PATTERN = re.compile(
    r"(?P<value>\d+(?:,\d{3})*(?:\.\d{1,2})?)\s*"
    r"(AED|د.إ|دينار|درهم)?",
    re.IGNORECASE,
)
ROOMS_PATTERN = re.compile(
    r"(?P<value>\d+)\s*(bedroom|bed|room|br|beds|غرفة|غرف|نوم)",
    re.IGNORECASE,
)

# Expected columns in the raw input
EXPECTED_COLUMNS: List[str] = [
    "description",
    "location",
    "property_type",
    "size_sqft",
    "price_aed",
    "bedrooms",
    "bathrooms",
]

CATEGORICAL_COLUMNS: List[str] = ["location"]

TARGET_COLUMN = "property_type"

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Normalise a single text field.

    Steps:
        1. Strip leading/trailing whitespace.
        2. Remove Arabic characters.
        3. Remove characters that are not alphanumeric, space, ``_``, ``/``,
           or ``-``.
        4. Coalesce whitespace runs.
        5. Lowercase.

    Args:
        text: Raw input string.

    Returns:
        Cleaned text.
    """
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = ARABIC_PATTERN.sub("", text)
    text = NON_ALPHANUM_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)
    return text.lower()


def clean_text_series(series: pd.Series) -> pd.Series:
    """Apply ``clean_text`` to every element of a Series."""
    return series.apply(clean_text)


# ---------------------------------------------------------------------------
# Numeric extraction helpers
# ---------------------------------------------------------------------------


def extract_size_sqft(series: pd.Series) -> pd.Series:
    """Extract numeric size (in sq. ft.) from a text Series.

    Falls back to the ``size_sqft`` column if available; otherwise parses
    the description.

    Returns:
        Series of float values (``NaN`` where not found).
    """
    def _extract(val: Any) -> Optional[float]:
        if pd.isna(val):
            return None
        match = SIZE_PATTERN.search(str(val))
        if match:
            raw = match.group("value").replace(",", "")
            return float(raw)
        return None
    return series.apply(_extract)


def extract_price(series: pd.Series) -> pd.Series:
    """Extract numeric price from a text Series.

    Returns:
        Series of float values (``NaN`` where not found).
    """
    def _extract(val: Any) -> Optional[float]:
        if pd.isna(val):
            return None
        match = PRICE_PATTERN.search(str(val))
        if match:
            raw = match.group("value").replace(",", "")
            return float(raw)
        return None
    return series.apply(_extract)


def extract_rooms(series: pd.Series) -> pd.Series:
    """Extract number of bedrooms / rooms from a text Series.

    Returns:
        Series of int values (``NaN`` where not found).
    """
    def _extract(val: Any) -> Optional[int]:
        if pd.isna(val):
            return None
        match = ROOMS_PATTERN.search(str(val))
        if match:
            return int(match.group("value"))
        return None
    return series.apply(_extract)


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------


def encode_categorical(
    df: pd.DataFrame,
    columns: List[str],
    encodings: Optional[Dict[str, Dict[str, int]]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    """Map categorical columns to integer codes.

    Args:
        df: Input DataFrame.
        columns: List of column names to encode.
        encodings: Pre-computed mappings (useful for consistent transform
            on unseen data). When ``None``, mappings are learned from data.

    Returns:
        ``(transformed_df, encodings)`` where *encodings* maps column names
        to ``{category: code}`` dictionaries.
    """
    df = df.copy()
    if encodings is None:
        encodings = {}
        for col in columns:
            if col in df.columns:
                codes = {cat: i for i, cat in enumerate(df[col].dropna().unique())}
                codes.setdefault("UNKNOWN", -1)
                encodings[col] = codes
                df[col] = df[col].map(codes).fillna(-1).astype(int)
    else:
        for col in columns:
            if col in df.columns and col in encodings:
                codes = encodings[col]
                df[col] = df[col].map(codes).fillna(-1).astype(int)
    return df, encodings


# ---------------------------------------------------------------------------
# Missing value handling
# ---------------------------------------------------------------------------


def handle_missing_values(
    df: pd.DataFrame,
    numeric_strategy: str = "median",
    categorical_fill: str = "UNKNOWN",
) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    Args:
        df: Input DataFrame.
        numeric_strategy: ``'mean'``, ``'median'``, or ``'mode'``.
        categorical_fill: Constant fill value for categorical columns.

    Returns:
        DataFrame with missing values imputed.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    if numeric_strategy == "mean":
        fill_values = df[numeric_cols].mean()
    elif numeric_strategy == "median":
        fill_values = df[numeric_cols].median()
    elif numeric_strategy == "mode":
        fill_values = df[numeric_cols].mode().iloc[0]
    else:
        raise ValueError(f"Unknown numeric_strategy: {numeric_strategy}")

    df[numeric_cols] = df[numeric_cols].fillna(fill_values)
    for col in df.columns:
        if col not in numeric_cols:
            df[col] = df[col].fillna(categorical_fill)
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional features from existing columns.

    Currently adds:
        - ``price_per_sqft``: price_aed / size_sqft
        - ``log_price``: log-transform of price
        - ``has_description``: binary indicator

    Args:
        df: Cleaned DataFrame.

    Returns:
        DataFrame with extra columns.
    """
    df = df.copy()
    if "price_aed" in df.columns and "size_sqft" in df.columns:
        size = pd.to_numeric(df["size_sqft"], errors="coerce").fillna(0)
        price = pd.to_numeric(df["price_aed"], errors="coerce").fillna(0)
        df["price_per_sqft"] = np.where(size > 0, price / size, 0.0)
    if "price_aed" in df.columns:
        price = pd.to_numeric(df["price_aed"], errors="coerce").fillna(0)
        df["log_price"] = np.log1p(price.clip(lower=0))
    if "description" in df.columns:
        df["has_description"] = df["description"].str.len() > 0
    return df


# ---------------------------------------------------------------------------
# Main preprocessing pipeline
# ---------------------------------------------------------------------------


def preprocess_property_data(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    test_size: float = 0.2,
    random_state: int = 42,
    encodings: Optional[Dict[str, Dict[str, int]]] = None,
    return_encodings: bool = False,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Dict[str, int]]],
]:
    """End-to-end preprocessing of UAE commercial property data.

    Steps:
        1. Load raw CSV.
        2. Clean text fields (description, location).
        3. Extract numeric features (size, price, rooms) from text.
        4. Encode categorical variables.
        5. Impute missing values.
        6. Engineer additional features.
        7. Perform a train/test split.

    Args:
        input_path: Path to raw CSV file.
        output_path: Path where the (train + test) cleaned data is saved.
        test_size: Fraction of data reserved for testing (default 0.2).
        random_state: Seed for reproducible splits.
        encodings: Pre-existing category → code mappings (optional).
        return_encodings: If ``True``, also return the encodings dict.

    Returns:
        ``X_train, X_test, y_train, y_test`` or the same plus encodings.

    Raises:
        FileNotFoundError: If *input_path* does not exist.
        ValueError: If the required columns are missing.
    """
    logger.info("Starting preprocessing — input: %s, output: %s", input_path, output_path)

    # Load
    df = load_dataframe(input_path)

    # Validate columns
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        logger.warning("Missing expected columns: %s. Proceeding with available data.", missing_cols)

    # --- Text cleaning ---
    if "description" in df.columns:
        df["description"] = clean_text_series(df["description"])
    if "location" in df.columns:
        df["location_clean"] = clean_text_series(df["location"])

    # --- Numeric extraction from text ---
    # If numeric columns are already present, use them; otherwise try to parse
    if "size_sqft" not in df.columns or df["size_sqft"].isna().all():
        df["size_sqft"] = extract_size_sqft(df.get("description", pd.Series(dtype=str)))
    if "price_aed" not in df.columns or df["price_aed"].isna().all():
        df["price_aed"] = extract_price(df.get("description", pd.Series(dtype=str)))
    if "bedrooms" not in df.columns or df["bedrooms"].isna().all():
        df["bedrooms"] = extract_rooms(df.get("description", pd.Series(dtype=str)))

    # --- Categorical encoding ---
    cats_to_encode = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    df, learned_encodings = encode_categorical(df, cats_to_encode, encodings)
    if encodings is None:
        encodings = learned_encodings

    # --- Missing values ---
    df = handle_missing_values(df)

    # --- Feature engineering ---
    df = engineer_features(df)

    # Separate features / target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in data. "
            f"Available columns: {list(df.columns)}"
        )
    y = df[TARGET_COLUMN].copy()

    feature_cols = [c for c in df.columns if c != TARGET_COLUMN]
    X = df[feature_cols].copy()

    # Train/test split (fall back to non-stratified if any class has too few samples)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
    except ValueError:
        logger.warning("Stratified split failed — falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train
    train_df["split"] = "train"
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test
    test_df["split"] = "test"

    combined = pd.concat([train_df, test_df], ignore_index=True)
    save_dataframe(combined, output_path)

    logger.info(
        "Preprocessing complete — train: %d, test: %d, features: %d",
        len(X_train), len(X_test), X_train.shape[1],
    )

    if return_encodings:
        return X_train, X_test, y_train, y_test, encodings
    return X_train, X_test, y_train, y_test
