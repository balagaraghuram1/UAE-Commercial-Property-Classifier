# UAE Commercial Property Classifier

The UAE Commercial Property Classifier is a machine learning project designed to classify commercial properties based on textual data. This project handles data preprocessing, model training, inference, and evaluation with models like Logistic Regression, Random Forest, and BERT.

---

## Project Structure

1. **Data Directory**:
   - `data/raw/`: Contains raw property data (e.g., CSV files with descriptions, locations, etc.).
   - `data/processed/`: Preprocessed data ready for modeling.
   - `data/results/`: Classification results.

2. **Models Directory**:
   - `models/`: Stores trained models:
     - `property_classifier.pkl`: Logistic Regression or Random Forest model.
     - `embeddings/`: Pre-trained embeddings (e.g., Word2Vec, GloVe).

3. **Notebooks Directory**:
   - `notebooks/`: Jupyter notebooks for:
     - **eda.ipynb**: Exploratory Data Analysis.
     - **training.ipynb**: Model training and evaluation.
     - **classification.ipynb**: Running classifications on test data.

4. **Source Code Directory**:
   - `src/`: Core scripts for the pipeline:
     - `preprocess.py`: Cleans and prepares data.
     - `train.py`: Trains classification models.
     - `inference.py`: Classifies new property data.
     - `evaluate.py`: Computes evaluation metrics.
     - `utils.py`: Helper functions for data handling.

5. **Tests Directory**:
   - `tests/`: Unit tests for preprocessing, training, and pipeline.

6. **Other Files**:
   - `.gitignore`: Specifies files/folders to exclude from version control.
   - `LICENSE`: Project license.
   - `README.md`: Overview of the project (this file).
   - `requirements.txt`: Python dependencies.
   - `main.py`: Orchestrates the pipeline via CLI.

---

## How to Use

1. **Setup**:
   - Clone the repository:
     ```bash
     git clone https://github.com/your-username/UAE-Commercial-Property-Classifier.git
     cd UAE-Commercial-Property-Classifier
     ```
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Commands**:
   - Preprocess Data:
     ```bash
     python main.py preprocess --input data/raw/property_data.csv --output data/processed/cleaned_data.csv
     ```
   - Train a Model:
     ```bash
     python main.py train --model logistic_regression
     ```
   - Perform Inference:
     ```bash
     python main.py inference --model_path models/property_classifier.pkl --input_path data/processed/cleaned_data.csv --output_path data/results/classification_results.csv
     ```
   - Evaluate a Model:
     ```bash
     python main.py evaluate --model_path models/property_classifier.pkl --metrics accuracy precision recall f1
     ```

---

## Dependencies

Install required packages with:
```bash
pip install -r requirements.txt
