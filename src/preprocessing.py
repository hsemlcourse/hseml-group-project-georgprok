from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "traffic_volume"

NUMERIC_FEATURES = [
    "temp",
    "rain_1h",
    "snow_1h",
    "clouds_all",
    "hour",
    "day_of_week",
    "month",
    "year",
    "is_weekend",
    "is_rush_hour",
]

CATEGORICAL_FEATURES = ["holiday", "weather_main", "weather_description"]


def load_traffic_data(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["date_time"] = pd.to_datetime(data["date_time"], errors="coerce")
    data = data.dropna(subset=["date_time", TARGET])
    data["holiday"] = data["holiday"].fillna("None")
    data = data.drop_duplicates()
    data.loc[data["temp"] <= 1, "temp"] = np.nan
    data = data.sort_values("date_time").reset_index(drop=True)
    return data


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["hour"] = data["date_time"].dt.hour
    data["day_of_week"] = data["date_time"].dt.dayofweek
    data["month"] = data["date_time"].dt.month
    data["year"] = data["date_time"].dt.year
    data["is_weekend"] = data["day_of_week"].isin([5, 6]).astype(int)
    data["is_rush_hour"] = data["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)
    return data


def time_based_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.sort_values("date_time").reset_index(drop=True)
    train = data[data["date_time"] < "2017-01-01"].copy()
    validation = data[(data["date_time"] >= "2017-01-01") & (data["date_time"] < "2018-01-01")].copy()
    test = data[data["date_time"] >= "2018-01-01"].copy()
    return train, validation, test


def _make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )
