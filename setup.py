"""setup.py — pip installable package for the UAE Commercial Property Classifier."""

from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup

HERE = Path(__file__).resolve().parent

# Load the README for the long description
long_description = (HERE / "README.md").read_text(encoding="utf-8")

# Load the requirements
install_requires = [
    "pandas>=2.0.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",
    "joblib>=1.3.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.13.0",
    "python-dotenv>=1.0.0",
]

# Optional extras
extras_require = {
    "xgboost": ["xgboost>=2.0.0"],
    "bert": ["transformers>=4.35.0", "torch>=2.1.0"],
    "dev": [
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
        "xgboost>=2.0.0",
    ],
    "all": [
        "xgboost>=2.0.0",
        "transformers>=4.35.0",
        "torch>=2.1.0",
        "pytest>=8.0.0",
        "pytest-cov>=4.1.0",
    ],
}

setup(
    name="uae-property-classifier",
    version="1.0.0",
    description="Classify UAE commercial real estate properties from text descriptions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="balaga raghuram",
    author_email="team@example.com",
    url="https://github.com/your-username/UAE-Commercial-Property-Classifier",
    license="MIT",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests", "tests.*", "data", "notebooks"]),
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "uae-classifier=main:main",
        ],
    },
)
