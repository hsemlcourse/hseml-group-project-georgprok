import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42


def build_model_pipeline(estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


def get_model_candidates() -> dict[str, Pipeline]:
    return {
        "Dummy mean": build_model_pipeline(DummyRegressor(strategy="mean")),
        "Ridge": build_model_pipeline(Ridge()),
        "Lasso": build_model_pipeline(Lasso()),
        "KNN": build_model_pipeline(KNeighborsRegressor(n_jobs=1)),
        "DecisionTree": build_model_pipeline(DecisionTreeRegressor(random_state=RANDOM_STATE)),
        "RandomForest": build_model_pipeline(
            RandomForestRegressor(
                n_estimators=80,
                max_depth=20,
                min_samples_leaf=2,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "ExtraTrees": build_model_pipeline(
            ExtraTreesRegressor(
                n_estimators=80,
                random_state=RANDOM_STATE,
                n_jobs=1,
            )
        ),
        "HistGradientBoosting": build_model_pipeline(
            HistGradientBoostingRegressor(
                max_iter=200,
                learning_rate=0.08,
                random_state=RANDOM_STATE,
            )
        ),
    }


# Hyperparameter grids for RandomizedSearchCV tuning of tree-based models
PARAM_GRIDS: dict[str, dict] = {
    "RandomForest": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "ExtraTrees": {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [10, 20, None],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"],
    },
    "HistGradientBoosting": {
        "model__max_iter": [100, 200, 300],
        "model__learning_rate": [0.05, 0.08, 0.1, 0.15],
        "model__max_depth": [None, 5, 8],
        "model__l2_regularization": [0.0, 0.1, 1.0],
        "model__min_samples_leaf": [20, 40, 60],
    },
}


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def evaluate_model(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    return regression_metrics(y, model.predict(x))


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def save_metrics(metrics: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
