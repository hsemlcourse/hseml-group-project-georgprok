import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.modeling import evaluate_model, get_model_candidates, regression_metrics
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_time_features,
    build_preprocessor,
    clean_traffic_data,
    time_based_split,
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@pytest.fixture
def raw_sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "holiday": [None, "Christmas Day", None, None, None],
            "temp": [290.0, 0.0, 280.0, 275.0, 295.0],
            "rain_1h": [0.0, 0.0, 1.2, 0.0, 0.0],
            "snow_1h": [0.0, 0.0, 0.0, 0.0, 0.0],
            "clouds_all": [40, 75, 90, 0, 20],
            "weather_main": ["Clouds", "Clear", "Rain", "Clear", "Clouds"],
            "weather_description": ["scattered clouds", "sky is clear", "light rain", "sky is clear", "few clouds"],
            "date_time": [
                "2015-01-01 09:00:00",
                "2015-01-01 10:00:00",
                "2015-01-01 11:00:00",
                "2015-01-01 12:00:00",
                "2015-01-01 13:00:00",
            ],
            "traffic_volume": [3000, 4000, 3500, 2800, 3200],
        }
    )


@pytest.fixture
def clean_featured(raw_sample: pd.DataFrame) -> pd.DataFrame:
    return add_time_features(clean_traffic_data(raw_sample))


# --- clean_traffic_data ---


def test_clean_fills_holiday_nan(raw_sample: pd.DataFrame) -> None:
    cleaned = clean_traffic_data(raw_sample)
    assert (cleaned["holiday"] == "None").any()
    assert cleaned["holiday"].isna().sum() == 0


def test_clean_marks_low_temp_as_nan(raw_sample: pd.DataFrame) -> None:
    cleaned = clean_traffic_data(raw_sample)
    assert cleaned["temp"].isna().sum() >= 1


def test_clean_parses_datetime(raw_sample: pd.DataFrame) -> None:
    cleaned = clean_traffic_data(raw_sample)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date_time"])


def test_clean_sorted_by_datetime(raw_sample: pd.DataFrame) -> None:
    cleaned = clean_traffic_data(raw_sample)
    assert cleaned["date_time"].is_monotonic_increasing


# --- add_time_features ---


def test_time_features_added(clean_featured: pd.DataFrame) -> None:
    for col in ["hour", "day_of_week", "month", "year", "is_weekend", "is_rush_hour"]:
        assert col in clean_featured.columns


def test_is_weekend_binary(clean_featured: pd.DataFrame) -> None:
    assert set(clean_featured["is_weekend"].unique()).issubset({0, 1})


def test_is_rush_hour_binary(clean_featured: pd.DataFrame) -> None:
    assert set(clean_featured["is_rush_hour"].unique()).issubset({0, 1})


# --- time_based_split ---


@pytest.fixture
def large_df() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2018-06-01", freq="h")
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "date_time": dates,
            "traffic_volume": rng.integers(0, 7000, len(dates)),
            "holiday": "None",
            "temp": 280.0,
            "rain_1h": 0.0,
            "snow_1h": 0.0,
            "clouds_all": 40,
            "weather_main": "Clear",
            "weather_description": "sky is clear",
        }
    )


def test_split_no_temporal_overlap(large_df: pd.DataFrame) -> None:
    train, valid, test = time_based_split(large_df)
    assert train["date_time"].max() < valid["date_time"].min()
    assert valid["date_time"].max() < test["date_time"].min()


def test_split_covers_all_rows(large_df: pd.DataFrame) -> None:
    train, valid, test = time_based_split(large_df)
    assert len(train) + len(valid) + len(test) == len(large_df)


def test_split_boundaries(large_df: pd.DataFrame) -> None:
    train, valid, test = time_based_split(large_df)
    assert train["date_time"].max() < pd.Timestamp("2017-01-01")
    assert valid["date_time"].min() >= pd.Timestamp("2017-01-01")
    assert test["date_time"].min() >= pd.Timestamp("2018-01-01")


# --- Feature / target isolation ---


def test_target_not_in_feature_columns() -> None:
    assert TARGET not in FEATURE_COLUMNS


def test_date_time_not_in_feature_columns() -> None:
    assert "date_time" not in FEATURE_COLUMNS


# --- build_preprocessor ---


def test_preprocessor_no_nan_after_fit(clean_featured: pd.DataFrame) -> None:
    X = clean_featured[FEATURE_COLUMNS]
    y = clean_featured[TARGET]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X, y)
    assert not np.isnan(X_transformed).any()


def test_preprocessor_output_shape(clean_featured: pd.DataFrame) -> None:
    X = clean_featured[FEATURE_COLUMNS]
    preprocessor = build_preprocessor()
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)


# --- regression_metrics ---


def test_metrics_perfect_predictions() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, np.array([1.0, 2.0, 3.0]))
    assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["MAE"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["R2"] == pytest.approx(1.0)


def test_metrics_keys() -> None:
    y = pd.Series([1.0, 2.0, 3.0])
    metrics = regression_metrics(y, np.array([1.0, 2.5, 3.0]))
    assert set(metrics.keys()) == {"RMSE", "MAE", "R2"}


# --- Model pipeline (smoke test with fast model) ---


def test_ridge_pipeline_fit_predict(clean_featured: pd.DataFrame) -> None:
    X = clean_featured[FEATURE_COLUMNS]
    y = clean_featured[TARGET]
    pipeline: Pipeline = get_model_candidates()["Ridge"]
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(y)
    assert not np.isnan(preds).any()


def test_evaluate_model_returns_metrics(clean_featured: pd.DataFrame) -> None:
    X = clean_featured[FEATURE_COLUMNS]
    y = clean_featured[TARGET]
    pipeline = get_model_candidates()["Ridge"]
    pipeline.fit(X, y)
    metrics = evaluate_model(pipeline, X, y)
    assert "RMSE" in metrics
    assert metrics["RMSE"] >= 0
