"""Feature definitions and the train/validation/test split.

The model is deliberately limited to information that exists before any
compilation action starts:

  * the dominant directory and file type of the commit
  * the number of packages loaded and targets configured at the end of
    Bazel's analysis phase
  * the total number of actions in the build graph

`actions_executed` is kept in the processed CSV only for an upper-bound
reference experiment.  It is not in ANALYSIS_NUM_COLS and is never passed to
the deployed model.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_COLS = ["prefix", "type", "prefix_x_type"]

ANALYSIS_NUM_COLS = [
    "log_total_actions",
    "log_targets_configured",
    "log_packages_loaded",
    "actions_per_target",
    "targets_per_package",
    "is_cached",
    "is_build_file",
    "is_core_src",
    "is_test",
]

TARGET = "cpu_time_ms"
RISK_TARGET = "risk_tier"
RISK_LABELS = ["Low", "Medium", "High"]

FEATURE_COLS = CAT_COLS + ANALYSIS_NUM_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the derived analysis-phase features used by the model."""
    out = df.copy()
    out["log_total_actions"] = np.log1p(out["total_actions"])
    out["log_targets_configured"] = np.log1p(out["targets_configured"])
    out["log_packages_loaded"] = np.log1p(out["packages_loaded"])
    out["actions_per_target"] = out["total_actions"] / (out["targets_configured"] + 1)
    out["targets_per_package"] = out["targets_configured"] / (out["packages_loaded"] + 1)
    out["is_cached"] = (
        (out["targets_configured"] == 0) & (out["total_actions"] <= 2)
    ).astype(int)
    return out


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the same 300 / 100 / 100 chronological split as the Stanford paper."""
    if len(df) < 500:
        raise ValueError(f"Expected at least 500 rows, found {len(df)}")
    return df.iloc[:300].copy(), df.iloc[300:400].copy(), df.iloc[400:500].copy()


def build_preprocessor(extra_numeric: Iterable[str] = ()) -> ColumnTransformer:
    """One-hot encode commit metadata and scale the numeric analysis features."""
    numeric = ANALYSIS_NUM_COLS + list(extra_numeric)
    categorical = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numerical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", categorical, CAT_COLS),
            ("num", numerical, numeric),
        ],
        remainder="drop",
    )


def build_baseline_preprocessor() -> ColumnTransformer:
    """The Stanford baseline only sees the two commit metadata fields and their cross."""
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CAT_COLS,
            )
        ],
        remainder="drop",
    )


def make_input_frame(
    prefix: str,
    file_type: str,
    packages_loaded: float,
    targets_configured: float,
    total_actions: float,
) -> pd.DataFrame:
    """Build a one-row frame for the CLI and the notebook demo."""
    base = pd.DataFrame(
        [
            {
                "prefix": prefix,
                "type": file_type,
                "prefix_x_type": f"{prefix}_x_{file_type}",
                "packages_loaded": float(packages_loaded),
                "targets_configured": float(targets_configured),
                "total_actions": float(total_actions),
                "is_build_file": int(file_type == "Starlark"),
                "is_core_src": int(prefix == "src/main"),
                "is_test": int(prefix == "src/test"),
            }
        ]
    )
    return add_features(base)
