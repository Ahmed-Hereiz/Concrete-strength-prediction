# Concrete Strength Prediction

This project provides a machine learning solution for predicting the compressive strength of concrete mixtures based on their composition and age. It includes:

- **Model training scripts** for building and evaluating regression models.
- **A modern desktop GUI** for interactive predictions.
- **A FastAPI backend** for web-based or API access.

## Project Structure

```
concrete-strenght-prediction/
│
├── app.py                # FastAPI backend for predictions
├── gui.py                # PyQt5 desktop GUI application
├── model-training.py     # Model training and evaluation script
├── preprocessing.py      # Feature engineering and preprocessing functions
├── requirements.txt      # Python dependencies
├── saved/
│   ├── models/           # Trained model files (.joblib)
│   ├── assets/           # Evaluation plots, SHAP explanations, etc.
│   └── preprocessing/    # Encoders (if any)
├── data/
│   └── Concrete_Data.xls # Raw dataset
└── static/
    └── index.html        # Frontend for FastAPI (if used)
```

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ahmed-Hereiz/Concrete-strength-prediction.git
   cd concrete-strenght-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the dataset:**
   - Place `Concrete_Data.xls` in the `data/` directory.

## Model Training

To train and evaluate the models:

```bash
python model-training.py
```

- This will preprocess the data, train several regression models, save them in `saved/models/`, and generate evaluation plots and SHAP explanations in `saved/assets/`.

## Using the Desktop GUI

The GUI allows interactive prediction of concrete strength:

```bash
python gui.py
```

- Adjust mixture parameters using sliders.
- Select a prediction model.
- View predicted strength and feature details.

**Requirements:** PyQt5 (installed via `requirements.txt`).

## Using the FastAPI Backend

Start the API server:

```bash
python app.py
```

- The API will be available at `http://localhost:8000`.
- The main page (if `static/index.html` exists) is served at `/`.
- Prediction endpoint: `POST /predict`


## Requirements

- Python 3.7+
- See `requirements.txt` for all dependencies (scikit-learn, pandas, numpy, PyQt5, fastapi, uvicorn, lightgbm, shap, etc.)

## Notes

- Models must be trained before using the GUI or API (`saved/models/` must exist).
- Feature engineering is handled via `preprocessing.py`.
- SHAP explanations and evaluation plots are generated for model interpretability.
