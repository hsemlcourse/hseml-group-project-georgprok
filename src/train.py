"""End-to-end training script. Run with: python -m src.train"""

import os
from pathlib import Path

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit

from src.modeling import (
    PARAM_GRIDS,
    RANDOM_STATE,
    evaluate_model,
    get_model_candidates,
    save_metrics,
    save_model,
)
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    add_time_features,
    clean_traffic_data,
    load_traffic_data,
    time_based_split,
)

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Metro_Interstate_Traffic_Volume.csv"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
N_ITER_SEARCH = 10
CV_SPLITS = 3


def main() -> None:
    print("Loading and preprocessing data...")
    raw_df = load_traffic_data(DATA_PATH)
    df = add_time_features(clean_traffic_data(raw_df))
    train_df, valid_df, test_df = time_based_split(df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET]
    X_valid = valid_df[FEATURE_COLUMNS]
    y_valid = valid_df[TARGET]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET]

    print(f"Train: {len(train_df)} rows, Valid: {len(valid_df)} rows, Test: {len(test_df)} rows")

    # Step 1: Baseline — fit all candidates, evaluate on validation
    print("\n--- Baseline evaluation of all candidates ---")
    models = get_model_candidates()
    baseline_results = []
    fitted_models: dict = {}

    for name, pipeline in models.items():
        print(f"  Fitting {name}...")
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_valid, y_valid)
        baseline_results.append({"model": name, **metrics})
        fitted_models[name] = pipeline
        print(f"    RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")

    baseline_df = pd.DataFrame(baseline_results).sort_values("RMSE").reset_index(drop=True)
    print("\nBaseline ranking (by validation RMSE):")
    print(baseline_df[["model", "RMSE", "MAE", "R2"]].to_string(index=False))

    # Step 2: Tune tree-based models with RandomizedSearchCV + TimeSeriesSplit
    # TimeSeriesSplit respects temporal order — no data leakage into the past
    models_to_tune = [name for name in PARAM_GRIDS if name in models]
    print(f"\n--- Hyperparameter tuning for: {models_to_tune} ---")

    tscv = TimeSeriesSplit(n_splits=CV_SPLITS)
    tuned_fitted: dict = {}
    tuned_results = []
    tuned_best_params: dict = {}

    for name in models_to_tune:
        print(f"  Tuning {name} (n_iter={N_ITER_SEARCH}, cv={CV_SPLITS})...")
        base_pipeline = clone(fitted_models[name])
        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=PARAM_GRIDS[name],
            n_iter=N_ITER_SEARCH,
            scoring="neg_root_mean_squared_error",
            cv=tscv,
            random_state=RANDOM_STATE,
            n_jobs=1,
            refit=True,
        )
        search.fit(X_train, y_train)
        best_estimator = search.best_estimator_
        metrics = evaluate_model(best_estimator, X_valid, y_valid)
        tuned_fitted[name] = best_estimator
        tuned_results.append({"model": f"{name} (tuned)", **metrics})
        tuned_best_params[name] = search.best_params_
        print(f"    Best params: {search.best_params_}")
        print(f"    RMSE={metrics['RMSE']:.2f}, MAE={metrics['MAE']:.2f}, R2={metrics['R2']:.4f}")

    # Step 3: Select overall best by validation RMSE
    all_candidates = []
    for r in baseline_results:
        all_candidates.append(
            {"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"], "pipeline": fitted_models[r["model"]]}
        )
    for r in tuned_results:
        base_name = r["model"].replace(" (tuned)", "")
        all_candidates.append(
            {"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"], "pipeline": tuned_fitted[base_name]}
        )

    all_candidates.sort(key=lambda x: x["RMSE"])
    best = all_candidates[0]
    print(f"\nBest overall: {best['model']} (validation RMSE={best['RMSE']:.2f})")

    # Step 4: Retrain best on train+valid, evaluate on test
    print("\nRetraining best model on train+valid...")
    X_train_valid = pd.concat([X_train, X_valid], axis=0)
    y_train_valid = pd.concat([y_train, y_valid], axis=0)

    final_model = clone(best["pipeline"])
    final_model.fit(X_train_valid, y_train_valid)
    test_metrics = evaluate_model(final_model, X_test, y_test)
    print(f"Test metrics: RMSE={test_metrics['RMSE']:.2f}, MAE={test_metrics['MAE']:.2f}, R2={test_metrics['R2']:.4f}")

    # Step 5: Save model and metrics
    model_path = MODELS_DIR / "best_model.joblib"
    save_model(final_model, model_path)
    print(f"\nModel saved: {model_path}")

    output_metrics = {
        "best_model": best["model"],
        "validation": {"RMSE": best["RMSE"], "MAE": best["MAE"], "R2": best["R2"]},
        "test": test_metrics,
        "baseline_results": [{"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"]} for r in baseline_results],
        "tuned_results": [{"model": r["model"], "RMSE": r["RMSE"], "MAE": r["MAE"], "R2": r["R2"]} for r in tuned_results],
        "tuned_best_params": tuned_best_params,
    }
    metrics_path = MODELS_DIR / "metrics.json"
    save_metrics(output_metrics, metrics_path)
    print(f"Metrics saved: {metrics_path}")


if __name__ == "__main__":
    main()
