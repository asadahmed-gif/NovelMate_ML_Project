"""Shared helpers used by every NovelMate notebook: paths, preprocessing, data split, saving models."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "saved_models"
RAW_FILE = DATA_DIR / "books.csv"             # the Kaggle file you upload
CLEAN_FILE = DATA_DIR / "novels_clean.csv"    # written by Data_Processing.ipynb
INFO_FILE = MODEL_DIR / "model_info.joblib"   # model list + metrics, read by the web app

YEAR = "original_publication_year"
COUNTS = ["books_count", "ratings_count", "work_ratings_count", "work_text_reviews_count"]
LANG = "language_code"
TARGET = "average_rating"          # regression target
GROUP = "rating_group"             # classification target: Low / Medium / High
FEATURES = [YEAR] + COUNTS + [LANG]  # same order in training and in the API
CLASS_LABELS = ["Low", "Medium", "High"]


def make_preprocessor():
    """Impute missing values, log-transform and scale the counts, scale the year, one-hot encode the language."""
    counts = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),   # counts are heavily skewed
        ("scale", StandardScaler()),
    ])
    year = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    language = OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30, sparse_output=False)
    return ColumnTransformer([("counts", counts, COUNTS), ("year", year, [YEAR]), ("language", language, [LANG])])


def build_pipeline(estimator):
    """Preprocessing + model in one object, so the web app can feed raw values straight in."""
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def load_clean_data():
    if not CLEAN_FILE.exists():
        raise FileNotFoundError("data/novels_clean.csv not found. Run Data_Processing.ipynb first.")
    return pd.read_csv(CLEAN_FILE, keep_default_na=False, na_values=[""])


def split_data(df):
    """One fixed 80/20 split, identical in every notebook, so all models are compared fairly."""
    return train_test_split(df[FEATURES], df[TARGET], df[GROUP],
                            test_size=0.2, random_state=42, stratify=df[GROUP])


def load_info():
    if not INFO_FILE.exists():
        raise FileNotFoundError("saved_models/model_info.joblib not found. Run Data_Processing.ipynb first.")
    return joblib.load(INFO_FILE)


def register_model(task, key, name, pipeline, metrics):
    """Save the trained pipeline with joblib and add its metrics to model_info.joblib."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_DIR / f"{task}_{key}.joblib", compress=3)
    info = load_info()
    info["models"][task][key] = name
    entry = {"algorithm": key, "name": name, **{k: float(v) for k, v in metrics.items()}}
    rows = [m for m in info["metrics"][task] if m["algorithm"] != key]
    info["metrics"][task] = rows + [entry]
    joblib.dump(info, INFO_FILE)
    print(f"Saved saved_models/{task}_{key}.joblib and updated model_info.joblib")
