"""Shared configuration & feature metadata for the Heart Disease app."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "heart.csv"
MODEL_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "predictions.db"

NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
            "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
TARGET = "target"

# Human friendly metadata used to build the form, validate input and explain results
FEATURE_META = {
    "age":      {"label": "Age", "type": "number", "min": 1, "max": 120, "step": 1, "default": 54, "unit": "years",
                 "help": "Age of the patient."},
    "sex":      {"label": "Sex", "type": "select", "options": {1: "Male", 0: "Female"}, "default": 1,
                 "help": "Biological sex."},
    "cp":       {"label": "Chest pain type", "type": "select",
                 "options": {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"},
                 "default": 0, "help": "Type of chest pain experienced."},
    "trestbps": {"label": "Resting blood pressure", "type": "number", "min": 60, "max": 250, "step": 1,
                 "default": 130, "unit": "mm Hg", "help": "Resting blood pressure on admission."},
    "chol":     {"label": "Serum cholesterol", "type": "number", "min": 80, "max": 700, "step": 1,
                 "default": 240, "unit": "mg/dl", "help": "Total serum cholesterol."},
    "fbs":      {"label": "Fasting blood sugar > 120 mg/dl", "type": "select", "options": {0: "No", 1: "Yes"},
                 "default": 0, "help": "Whether fasting blood sugar exceeds 120 mg/dl."},
    "restecg":  {"label": "Resting ECG result", "type": "select",
                 "options": {0: "Normal", 1: "ST-T wave abnormality", 2: "Left ventricular hypertrophy"},
                 "default": 1, "help": "Resting electrocardiographic results."},
    "thalach":  {"label": "Maximum heart rate achieved", "type": "number", "min": 50, "max": 250, "step": 1,
                 "default": 150, "unit": "bpm", "help": "Peak heart rate during exercise test."},
    "exang":    {"label": "Exercise induced angina", "type": "select", "options": {0: "No", 1: "Yes"},
                 "default": 0, "help": "Chest pain triggered by exercise."},
    "oldpeak":  {"label": "ST depression (oldpeak)", "type": "number", "min": 0, "max": 10, "step": 0.1,
                 "default": 1.0, "unit": "mm", "help": "ST depression induced by exercise relative to rest."},
    "slope":    {"label": "Slope of peak-exercise ST segment", "type": "select",
                 "options": {0: "Upsloping", 1: "Flat", 2: "Downsloping"}, "default": 1,
                 "help": "Slope of the ST segment at peak exercise."},
    "ca":       {"label": "Major vessels colored by fluoroscopy", "type": "select",
                 "options": {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"}, "default": 0,
                 "help": "Number of major vessels (0-4) coloured by fluoroscopy."},
    "thal":     {"label": "Thalassemia", "type": "select",
                 "options": {0: "Unknown / null", 1: "Fixed defect", 2: "Normal", 3: "Reversible defect"},
                 "default": 2, "help": "Thallium stress test result."},
}

RISK_BANDS = [(0.30, "Low", "success"), (0.60, "Moderate", "warning"), (1.01, "High", "danger")]


def risk_band(p: float):
    for upper, name, color in RISK_BANDS:
        if p < upper:
            return name, color
    return "High", "danger"


# Label convention ---------------------------------------------------------------
# By default the app uses the `target` column exactly as it is in heart.csv
# (prediction 1 == target 1), matching the original notebook.
# NOTE: in this well-known dataset the clinical patterns (exercise angina, blocked vessels,
# reversible thal defect, low max heart rate) actually line up with target = 0, i.e. the
# file's label may be inverted relative to its documentation. If you want the app to treat
# 1 = "disease present" in the clinical sense, set  CARDIO_INVERT_TARGET=1  (or edit below)
# and run `python train.py` again.
import os
INVERT_TARGET = os.environ.get("CARDIO_INVERT_TARGET", "0") == "1"


def load_data(dedupe=True):
    """Load heart.csv (optionally de-duplicated); apply INVERT_TARGET if enabled."""
    import pandas as pd
    df = pd.read_csv(DATA_PATH)
    if dedupe:
        df = df.drop_duplicates().reset_index(drop=True)
    if INVERT_TARGET:
        df[TARGET] = 1 - df[TARGET]
    return df
