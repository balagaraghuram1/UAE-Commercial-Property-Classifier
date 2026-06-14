#!/usr/bin/env python3
"""
UAE Commercial Property Classifier — CLI entry point.

Orchestrates the machine learning pipeline with four subcommands:

    preprocess   → clean and prepare raw property data
    train        → train a classification model
    inference    → classify new property descriptions
    evaluate     → evaluate a trained model on labelled data

Usage:
    python main.py preprocess --input data/raw/property_data.csv
                              --output data/processed/cleaned_data.csv
    python main.py train --model logistic_regression
    python main.py inference --model_path models/property_classifier.pkl
                             --input_path data/processed/cleaned_data.csv
                             --output_path data/results/classification_results.csv
    python main.py evaluate --model_path models/property_classifier.pkl
                            --data_path data/processed/cleaned_data.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.preprocess import preprocess_property_data
from src.train import train_property_classifier
from src.inference import classify_properties
from src.evaluate import evaluate_classifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with consistent formatting.

    Args:
        verbose: If ``True``, set log level to ``DEBUG``.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


logger = logging.getLogger("main")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="UAE-Commercial-Property-Classifier",
        description=(
            "Classify UAE commercial property descriptions using "
            "Logistic Regression, Random Forest, XGBoost, or BERT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py preprocess --input data/raw/properties.csv --output data/processed/clean.csv\n"
            "  python main.py train --model random_forest\n"
            "  python main.py inference --model_path models/model.pkl --input_path data/processed/clean.csv --output_path data/results/out.csv\n"
            "  python main.py evaluate --model_path models/model.pkl --data_path data/processed/clean.csv\n"
        ),
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging.",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        required=True,
        help="Pipeline stage to execute.",
    )

    # ---- preprocess ----
    pp = subparsers.add_parser("preprocess", help="Clean and engineer features from raw property data.")
    pp.add_argument("--input", required=True, help="Path to raw CSV input.")
    pp.add_argument("--output", required=True, help="Path to save the cleaned CSV.")
    pp.add_argument("--test-size", type=float, default=0.2, help="Fraction for test split (default: 0.2).")
    pp.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")

    # ---- train ----
    tr = subparsers.add_parser("train", help="Train a property classification model.")
    tr.add_argument(
        "--model",
        choices=["logistic_regression", "random_forest", "xgboost", "bert"],
        default="logistic_regression",
        help="Model architecture (default: logistic_regression).",
    )
    tr.add_argument("--data", default="data/processed/cleaned_data.csv", help="Preprocessed CSV input.")
    tr.add_argument("--output", default="models/property_classifier.pkl", help="Model output path.")
    tr.add_argument("--metadata", default=None, help="Optional metadata JSON output path.")
    tr.add_argument("--no-grid-search", action="store_true", help="Skip GridSearchCV hyperparameter tuning.")
    tr.add_argument("--cv-folds", type=int, default=5, help="Cross-validation folds (default: 5).")

    # ---- inference ----
    inf = subparsers.add_parser("inference", help="Classify new property descriptions.")
    inf.add_argument("--model-path", required=True, help="Path to trained model artifact.")
    inf.add_argument("--input-path", required=True, help="Path to CSV with properties to classify.")
    inf.add_argument("--output-path", required=True, help="Destination CSV for predictions.")
    inf.add_argument("--text-column", default="description", help="Column holding description text.")
    inf.add_argument("--id-column", default=None, help="Optional ID column to preserve in output.")
    inf.add_argument("--batch-size", type=int, default=64, help="Batch size for BERT inference (default: 64).")

    # ---- evaluate ----
    ev = subparsers.add_parser("evaluate", help="Evaluate a trained model on labelled data.")
    ev.add_argument("--model-path", required=True, help="Path to trained model artifact.")
    ev.add_argument("--data-path", required=True, help="Path to labelled CSV (test set).")
    ev.add_argument("--output-dir", default="data/results/", help="Directory for metrics / plots.")
    ev.add_argument("--text-column", default="description", help="Column holding description text.")
    ev.add_argument("--target-column", default="property_type", help="Column with ground-truth labels.")

    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point — parse arguments and dispatch to the appropriate module.

    Args:
        argv: Command-line arguments (``None`` uses ``sys.argv[1:]``).

    Returns:
        Exit code (0 on success, 1 on error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    setup_logging(verbose=getattr(args, "verbose", False))
    logger.debug("Parsed arguments: %s", args)

    try:
        if args.command == "preprocess":
            preprocess_property_data(
                input_path=args.input,
                output_path=args.output,
                test_size=args.test_size,
                random_state=args.random_state,
            )

        elif args.command == "train":
            train_property_classifier(
                model_type=args.model,
                data_path=args.data,
                model_output_path=args.output,
                metadata_output_path=args.metadata,
                use_grid_search=not args.no_grid_search,
                cv_folds=args.cv_folds,
            )

        elif args.command == "inference":
            classify_properties(
                model_path=args.model_path,
                input_path=args.input_path,
                output_path=args.output_path,
                text_column=args.text_column,
                id_column=args.id_column,
                batch_size=args.batch_size,
            )

        elif args.command == "evaluate":
            evaluate_classifier(
                model_path=args.model_path,
                data_path=args.data_path,
                output_dir=args.output_dir,
                text_column=args.text_column,
                target_column=args.target_column,
            )

        else:
            parser.print_help()
            return 1

    except Exception:
        logger.exception("Pipeline failed")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
