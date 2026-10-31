# NovelMate: Novel Rating Prediction

A machine learning web application built with **Python, FastAPI, HTML, CSS and JavaScript**.
Enter a novel's statistics, choose a model (or compare all models) and see the prediction.

## Dataset
- **Goodbooks-10k** `books.csv` (Kaggle: https://www.kaggle.com/datasets/zygmunt/goodbooks-10k), about 10,000 novels
- **Features:** original publication year, number of editions, ratings count, work ratings count, work text reviews count, language code
- **Regression target:** `average_rating` (0 to 5)
- **Classification target:** rating group Low / Medium / High (three equal-sized groups of `average_rating`)

## Machine learning models
**Regression:** Linear Regression, Decision Tree, Random Forest, Support Vector Regression (SVR)

**Classification:** Logistic Regression, Decision Tree (Gini), Decision Tree (Entropy), Random Forest, Support Vector Machine (SVM), Naive Bayes

## Data preprocessing
Invalid values removed, missing values filled (median / "unknown"), log transform and scaling of the skewed counts, one-hot encoding of the language, fixed 80/20 train-test split.
The trained models are saved with **joblib** in `saved_models/`.

## How to run in Google Colab
1. Upload this whole folder to **Google Drive** as `My Drive/NovelMate_ML_Project` (unzip it first).
2. Open `Data_Processing.ipynb` in Colab (right click -> Open with -> Google Colaboratory) and choose `Runtime -> Run all`. `data/books.csv` is already included, so nothing needs uploading.
3. Open each model notebook and run all cells (any order): `Linear_Regression`, `Regression_Tree`, `Random_Forest`, `SVR`, `Logistic_Regression`, `Decision_Tree`, `SVM`, `Naive_Bayes`.
4. Open `Launch_App.ipynb` and run all cells. The web app appears inside the notebook, `REPORT.md` is written, and the project is downloaded as a zip.

Every notebook asks to connect Google Drive in its first cell, so the data and the saved models stay in the project folder.

## How to run on your own computer
```bash
pip install -r requirements.txt
python -m uvicorn app:app --reload
```
Open http://127.0.0.1:8000/ (API documentation: http://127.0.0.1:8000/docs). The `saved_models/` folder must already contain the trained models.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/info` | model list, metrics, form options |
| POST | `/predict/regression` | one regression model (`algorithm` + features) |
| POST | `/predict/classification` | one classification model |
| POST | `/compare/regression` | all trained regression models |
| POST | `/compare/classification` | all trained classification models |

## Project files
- `Data_Processing.ipynb`: loading, cleaning, exploring, saving the clean data
- `Linear_Regression.ipynb`, `Regression_Tree.ipynb`, `SVR.ipynb`: regression models
- `Logistic_Regression.ipynb`, `Decision_Tree.ipynb`, `SVM.ipynb`, `Naive_Bayes.ipynb`: classification models
- `Random_Forest.ipynb`: random forest for both tasks
- `Launch_App.ipynb`: starts the app in Colab, writes `REPORT.md`, downloads the zip
- `ml_utils.py`: shared preprocessing, split and model-saving helpers
- `app.py`: FastAPI backend
- `static/html/index.html`: frontend and JavaScript
- `static/css/style.css`: styling
- `saved_models/`: trained models (`.joblib`)
- `data/`: dataset
- `requirements.txt`: dependencies
