"""Train the early-phase regressor and the CI risk classifier.

Model selection uses the validation split only.  The test split is evaluated
once, after the model has been chosen, and every number in the report comes
from this script.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from data_loader import PROJECT_ROOT, load_dataset
from features import (
    ANALYSIS_NUM_COLS,
    CAT_COLS,
    FEATURE_COLS,
    RISK_LABELS,
    RISK_TARGET,
    TARGET,
    add_features,
    build_baseline_preprocessor,
    build_preprocessor,
    split_data,
)

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_pred = np.maximum(0.0, np.asarray(y_pred, dtype=float))
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _evaluate(model, preprocessor, splits: dict) -> dict:
    out = {}
    for name, frame in splits.items():
        x = preprocessor.transform(frame)
        out[name] = _metrics(frame[TARGET].values, model.predict(x))
    return out


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_dataset()
    df = add_features(raw)
    train, val, test = split_data(df)

    baseline_pre = build_baseline_preprocessor()
    x_train_base = baseline_pre.fit_transform(train)
    x_val_base = baseline_pre.transform(val)
    x_test_base = baseline_pre.transform(test)

    baseline = LinearRegression()
    baseline.fit(x_train_base, train[TARGET].values)
    baseline_metrics = {
        "train": _metrics(train[TARGET].values, baseline.predict(x_train_base)),
        "val": _metrics(val[TARGET].values, baseline.predict(x_val_base)),
        "test": _metrics(test[TARGET].values, baseline.predict(x_test_base)),
    }

    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train)
    x_val = preprocessor.transform(val)
    x_test = preprocessor.transform(test)

    candidates = {
        "Ridge Regression": Ridge(alpha=10.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=1,
            max_features=1.0,
            random_state=42,
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.08,
            max_depth=7,
            min_samples_leaf=10,
            random_state=42,
        ),
    }

    results = {}
    fitted = {}
    for name, model in candidates.items():
        model.fit(x_train, train[TARGET].values)
        fitted[name] = model
        results[name] = _evaluate(model, preprocessor, {"train": train, "val": val, "test": test})

    best_name = min(results, key=lambda name: results[name]["val"]["rmse"])
    best_model = fitted[best_name]

    # Reference experiment only: add actions actually executed, which is a
    # post-execution signal and therefore not usable for an early prediction.
    post_pre = build_preprocessor(extra_numeric=["actions_executed"])
    x_train_post = post_pre.fit_transform(train)
    x_val_post = post_pre.transform(val)
    x_test_post = post_pre.transform(test)
    post_model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.08,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42,
    )
    post_model.fit(x_train_post, train[TARGET].values)
    post_metrics = {
        "train": _metrics(train[TARGET].values, post_model.predict(x_train_post)),
        "val": _metrics(val[TARGET].values, post_model.predict(x_val_post)),
        "test": _metrics(test[TARGET].values, post_model.predict(x_test_post)),
    }

    # Risk classifier: balanced class weights so that High-risk builds are not
    # ignored by the majority Low class.
    risk_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    risk_model.fit(x_train, train[RISK_TARGET].values)
    risk_val_pred = risk_model.predict(x_val)
    risk_test_pred = risk_model.predict(x_test)

    risk_val = {
        "accuracy": float(accuracy_score(val[RISK_TARGET].values, risk_val_pred)),
        "macro_f1": float(f1_score(val[RISK_TARGET].values, risk_val_pred, average="macro")),
    }
    risk_test = {
        "accuracy": float(accuracy_score(test[RISK_TARGET].values, risk_test_pred)),
        "macro_f1": float(f1_score(test[RISK_TARGET].values, risk_test_pred, average="macro")),
        "confusion_matrix": confusion_matrix(
            test[RISK_TARGET].values, risk_test_pred, labels=RISK_LABELS
        ).tolist(),
        "classification_report": classification_report(
            test[RISK_TARGET].values,
            risk_test_pred,
            labels=RISK_LABELS,
            output_dict=True,
            zero_division=0,
        ),
    }

    matrix = np.asarray(risk_test["confusion_matrix"])
    risk_test["high_recall"] = float(matrix[2, 2] / matrix[2].sum()) if matrix[2].sum() else 0.0
    risk_test["high_to_low"] = int(matrix[2, 0])
    risk_test["high_to_medium"] = int(matrix[2, 1])
    risk_test["low_precision"] = float(matrix[0, 0] / matrix[:, 0].sum()) if matrix[:, 0].sum() else 0.0
    risk_test["low_recall"] = float(matrix[0, 0] / matrix[0].sum()) if matrix[0].sum() else 0.0

    # Save the deployment artifacts.
    joblib.dump(preprocessor, MODELS_DIR / "preprocessor.joblib")
    joblib.dump(best_model, MODELS_DIR / "regressor.joblib")
    joblib.dump(risk_model, MODELS_DIR / "risk_classifier.joblib")

    # Test predictions for plots and error analysis.
    test_pred = np.maximum(0.0, best_model.predict(x_test))
    base_pred = np.maximum(0.0, baseline.predict(x_test_base))
    probabilities = risk_model.predict_proba(x_test)
    class_order = list(risk_model.classes_)
    predictions = pd.DataFrame(
        {
            "commit_num": test["commit_num"].values,
            "prefix": test["prefix"].values,
            "type": test["type"].values,
            "actual_ms": test[TARGET].values,
            "baseline_ms": base_pred,
            "predicted_ms": test_pred,
            "residual_ms": test[TARGET].values - test_pred,
            "risk_actual": test[RISK_TARGET].values,
            "risk_predicted": risk_test_pred,
            "p_low": probabilities[:, class_order.index("Low")],
            "p_medium": probabilities[:, class_order.index("Medium")],
            "p_high": probabilities[:, class_order.index("High")],
        }
    )
    predictions.to_csv(REPORTS_DIR / "test_predictions.csv", index=False)

    rows = []
    rows.append({"model": "Stanford baseline (prefix and type)", **baseline_metrics["test"], "train_rmse": baseline_metrics["train"]["rmse"], "val_rmse": baseline_metrics["val"]["rmse"]})
    for name, metrics in results.items():
        rows.append(
            {
                "model": name,
                **metrics["test"],
                "train_rmse": metrics["train"]["rmse"],
                "val_rmse": metrics["val"]["rmse"],
            }
        )
    reg_table = pd.DataFrame(rows)[
        ["model", "train_rmse", "val_rmse", "rmse", "mae", "r2"]
    ].rename(columns={"rmse": "test_rmse", "mae": "test_mae", "r2": "test_r2"})
    reg_table.to_csv(REPORTS_DIR / "regression_metrics.csv", index=False)

    summary = {
        "dataset": {
            "builds": int(len(df)),
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
            "mean_cpu_ms": float(df[TARGET].mean()),
            "median_cpu_ms": float(df[TARGET].median()),
            "max_cpu_ms": float(df[TARGET].max()),
            "risk_counts": {k: int(v) for k, v in df[RISK_TARGET].value_counts().items()},
        },
        "baseline": baseline_metrics,
        "models": results,
        "selected_model": best_name,
        "selected_test": results[best_name]["test"],
        "postexec_reference": post_metrics,
        "risk_classifier": {
            "validation": risk_val,
            "test": risk_test,
        },
        "feature_columns": FEATURE_COLS,
        "thresholds_ms": {"low": 2500, "high": 10000},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    print("Selected model (validation RMSE):", best_name)
    print(reg_table.to_string(index=False))
    print("\nPost-execution reference (not deployable):")
    print(post_metrics["test"])
    print("\nRisk classifier test:")
    print(json.dumps({k: v for k, v in risk_test.items() if k != "classification_report"}, indent=2))


if __name__ == "__main__":
    main()
