# Mood Predictor

A modular machine learning project for emotion classification using SVM.

## Project Structure

- `src/`: Source code
  - `preprocess.py`: Text cleaning and preprocessing logic.
  - `train.py`: Model training with hyperparameter tuning (GridSearchCV).
  - `predict.py`: Interactive CLI for emotion prediction.
- `models/`: Directory where trained models and evaluation reports are stored.
- `requirements.txt`: Project dependencies.
- `text.csv`: Dataset used for training.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model:
   ```bash
   python src/train.py
   ```
   This will perform hyperparameter tuning and save the best model to the `models/` directory.

3. Predict emotions:
   ```bash
   python src/predict.py
   ```
   Follow the prompts to enter text and get emotion predictions.

## Emotions Supported

- sadness (0)
- joy (1)
- love (2)
- anger (3)
- fear (4)
- surprise (5)
