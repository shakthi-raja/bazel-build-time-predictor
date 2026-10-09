"""Command-line demo for the early-phase Bazel build predictor.

Example:
    python src/predict.py --prefix src/main --type JAVA \
        --packages-loaded 4 --targets-configured 1462 --total-actions 1992
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_loader import PROJECT_ROOT
from features import RISK_LABELS, make_input_frame

MODELS_DIR = PROJECT_ROOT / "models"
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "bazel_build_dataset.csv"

PREFIXES = [
    "bazelci/",
    "scripts/",
    "site/",
    "src/main",
    "src/test",
    "src/tools",
    "third_party/",
    "tools/",
]
FILE_TYPES = ["JAVA", "C/C++", "Starlark", "python"]


def load_artifacts(models_dir: Path = MODELS_DIR):
    paths = {
        "preprocessor": models_dir / "preprocessor.joblib",
        "regressor": models_dir / "regressor.joblib",
        "classifier": models_dir / "risk_classifier.joblib",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model files. Run `python src/train.py` first.\n" + "\n".join(missing)
        )
    return {name: joblib.load(path) for name, path in paths.items()}


def predict_build(
    prefix: str,
    file_type: str,
    packages_loaded: float,
    targets_configured: float,
    total_actions: float,
    artifacts: dict | None = None,
) -> dict:
    artifacts = artifacts or load_artifacts()
    row = make_input_frame(prefix, file_type, packages_loaded, targets_configured, total_actions)
    x = artifacts["preprocessor"].transform(row)
    predicted_ms = max(0.0, float(artifacts["regressor"].predict(x)[0]))
    probabilities = artifacts["classifier"].predict_proba(x)[0]
    classes = list(artifacts["classifier"].classes_)
    probability_map = {label: float(probabilities[classes.index(label)]) for label in RISK_LABELS}
    risk = max(probability_map, key=probability_map.get)

    if risk == "Low":
        action = "Small, cache-friendly build. Safe to run locally."
    elif risk == "Medium":
        action = "Standard build. Use a normal CI worker and keep the default timeout."
    else:
        action = "Large build. Use a high-memory worker and extend the timeout."

    return {
        "predicted_ms": predicted_ms,
        "risk_tier": risk,
        "probabilities": probability_map,
        "recommendation": action,
        "features": row.iloc[0].to_dict(),
    }


def _print_report(result: dict, prefix: str, file_type: str, actual: dict | None = None) -> None:
    probs = result["probabilities"]
    print()
    print("=" * 64)
    print(" BAZEL BUILD TIME AND RISK ESTIMATE")
    print("=" * 64)
    print(f" Commit profile       : {prefix} / {file_type}")
    print(
        f" Analysis phase       : {int(result['features']['packages_loaded'])} packages, "
        f"{int(result['features']['targets_configured'])} targets, "
        f"{int(result['features']['total_actions'])} actions"
    )
    print(
        f" Predicted CPU time   : {result['predicted_ms']:,.0f} ms "
        f"({result['predicted_ms'] / 1000:.2f} s)"
    )
    print(f" Risk tier            : {result['risk_tier']}")
    print(
        " Class probabilities  : "
        f"Low {probs['Low']:.1%} | Medium {probs['Medium']:.1%} | High {probs['High']:.1%}"
    )
    print(f" Recommended action   : {result['recommendation']}")
    if actual:
        print("-" * 64)
        print(
            f" Actual for this build: {actual['cpu_time_ms']:,.0f} ms, "
            f"actual tier {actual['risk_tier']}"
        )
    print("=" * 64)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Early-phase Bazel build time and risk predictor")
    parser.add_argument("--prefix", choices=PREFIXES, default="src/main")
    parser.add_argument("--type", dest="file_type", choices=FILE_TYPES, default="JAVA")
    parser.add_argument("--packages-loaded", type=float, default=4)
    parser.add_argument("--targets-configured", type=float, default=1462)
    parser.add_argument("--total-actions", type=float, default=1992)
    parser.add_argument(
        "--sample",
        type=int,
        help="Use the analysis features of a real commit number from the processed dataset.",
    )
    args = parser.parse_args()

    actual = None
    if args.sample is not None:
        data = pd.read_csv(DATA_CSV)
        match = data[data["commit_num"] == args.sample]
        if match.empty:
            parser.error(f"Commit {args.sample} is not in the dataset (1 to 500).")
        row = match.iloc[0]
        prefix, file_type = row["prefix"], row["type"]
        packages, targets, actions = (
            row["packages_loaded"],
            row["targets_configured"],
            row["total_actions"],
        )
        actual = {"cpu_time_ms": float(row["cpu_time_ms"]), "risk_tier": row["risk_tier"]}
    else:
        prefix, file_type = args.prefix, args.file_type
        packages, targets, actions = (
            args.packages_loaded,
            args.targets_configured,
            args.total_actions,
        )

    result = predict_build(prefix, file_type, packages, targets, actions)
    _print_report(result, prefix, file_type, actual)


if __name__ == "__main__":
    main()
