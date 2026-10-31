"""NovelMate - FastAPI backend: serves the web page and the prediction API."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "saved_models"
STATIC_DIR = BASE_DIR / "static"
INFO_PATH = MODEL_DIR / "model_info.joblib"

STORE = {"info": None, "mtime": None, "regression": {}, "classification": {}}


def load_models():
    """Load model_info and all saved models. Reloads automatically when a notebook saves a new model."""
    if not INFO_PATH.exists():
        return False
    mtime = INFO_PATH.stat().st_mtime
    if STORE["info"] is not None and STORE["mtime"] == mtime:
        return True
    info = joblib.load(INFO_PATH)
    for task in ("regression", "classification"):
        STORE[task] = {key: joblib.load(MODEL_DIR / f"{task}_{key}.joblib") for key in info["models"][task]}
    STORE["info"], STORE["mtime"] = info, mtime
    return True


app = FastAPI(title="NovelMate - Novel Rating Prediction")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class BookFeatures(BaseModel):
    original_publication_year: float = Field(..., ge=-4000, le=2100)
    books_count: float = Field(..., ge=0)
    ratings_count: float = Field(..., ge=0)
    work_ratings_count: float = Field(..., ge=0)
    work_text_reviews_count: float = Field(..., ge=0)
    language_code: str = Field("unknown", max_length=20)

    model_config = {"json_schema_extra": {"example": {
        "original_publication_year": 2000, "books_count": 50, "ratings_count": 50000,
        "work_ratings_count": 60000, "work_text_reviews_count": 3000, "language_code": "eng"}}}


class PredictRequest(BookFeatures):
    algorithm: str = Field(..., description="Key of the model, e.g. random_forest")


def require_models(task=None):
    if not load_models():
        raise HTTPException(status_code=503, detail="No trained models found. Run Data_Processing.ipynb and the model notebooks first.")
    if task and not STORE[task]:
        raise HTTPException(status_code=503, detail=f"No {task} model has been trained yet. Run the {task} notebooks first.")


def get_model(task: str, key: str):
    require_models(task)
    if key not in STORE[task]:
        raise HTTPException(status_code=404, detail=f"Unknown {task} algorithm '{key}'. Choose from: {list(STORE[task])}")
    return STORE[task][key]


def to_frame(book: BookFeatures) -> pd.DataFrame:
    values = book.model_dump()
    columns = STORE["info"]["features"]
    return pd.DataFrame([{c: values[c] for c in columns}], columns=columns)


def regression_result(key: str, row: pd.DataFrame) -> dict:
    value = float(get_model("regression", key).predict(row)[0])
    value = min(5.0, max(0.0, value))
    return {"algorithm": key, "name": STORE["info"]["models"]["regression"][key],
            "predicted_average_rating": round(value, 2)}


def classification_result(key: str, row: pd.DataFrame) -> dict:
    model = get_model("classification", key)
    label = str(model.predict(row)[0])
    probabilities = None
    if hasattr(model, "predict_proba"):             # SVM has no probabilities by default
        probabilities = {str(c): round(float(p), 4) for c, p in zip(model.classes_, model.predict_proba(row)[0])}
    return {"algorithm": key, "name": STORE["info"]["models"]["classification"][key],
            "category": label, "range": STORE["info"]["class_ranges"][label], "probabilities": probabilities}


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "html" / "index.html")


@app.get("/api/info")
def api_info():
    require_models()
    return STORE["info"]


@app.post("/predict/regression")
def predict_regression(request: PredictRequest):
    return regression_result(request.algorithm, to_frame(request))


@app.post("/predict/classification")
def predict_classification(request: PredictRequest):
    return classification_result(request.algorithm, to_frame(request))


@app.post("/compare/regression")
def compare_regression(book: BookFeatures):
    require_models("regression")
    row = to_frame(book)
    return {"results": [regression_result(key, row) for key in STORE["regression"]]}


@app.post("/compare/classification")
def compare_classification(book: BookFeatures):
    require_models("classification")
    row = to_frame(book)
    return {"results": [classification_result(key, row) for key in STORE["classification"]]}
