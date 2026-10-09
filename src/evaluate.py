"""Create the figures used in the report, slides and dashboard.

All evaluation figures are produced from the saved test predictions and the
selected model.  Permutation importance is computed on the validation split,
so the test split is not reused for model interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance

from data_loader import PROJECT_ROOT, load_dataset
from features import ANALYSIS_NUM_COLS, CAT_COLS, TARGET, add_features, split_data

matplotlib.use("Agg")

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

INK = "#1B2A41"
BLUE = "#1F4E79"
TEAL = "#2A9D8F"
AMBER = "#E9C46A"
RED = "#E76F51"
GRAY = "#8D99AE"
GRID = "#E3E8EF"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 1.0,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
        "savefig.bbox": "tight",
    }
)

COLORS = {"Low": TEAL, "Medium": AMBER, "High": RED}


def _clean_feature(name: str) -> str:
    if name.startswith("cat__prefix_x_type_"):
        return "prefix x type: " + name.replace("cat__prefix_x_type_", "")
    if name.startswith("cat__prefix_"):
        return "prefix: " + name.replace("cat__prefix_", "")
    if name.startswith("cat__type_"):
        return "type: " + name.replace("cat__type_", "")
    if name.startswith("num__"):
        label = name.replace("num__", "")
        labels = {
            "log_total_actions": "log action count",
            "log_targets_configured": "log targets configured",
            "log_packages_loaded": "log packages loaded",
            "actions_per_target": "actions per target",
            "targets_per_package": "targets per package",
            "is_cached": "cached build flag",
            "is_build_file": "Starlark change flag",
            "is_core_src": "src/main flag",
            "is_test": "src/test flag",
        }
        return labels.get(label, label.replace("_", " "))
    return name


def target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    bins = np.arange(0, 30_001, 1_000)
    ax.hist(df[TARGET].clip(upper=30_000), bins=bins, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvspan(0, 2_500, color=TEAL, alpha=0.10)
    ax.axvspan(2_500, 10_000, color=AMBER, alpha=0.12)
    ax.axvspan(10_000, 30_000, color=RED, alpha=0.10)
    ax.set_xlim(0, 30_000)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_xlabel("CPU build time (ms)")
    ax.set_ylabel("Number of builds")
    ax.set_title("Build CPU time and the three SLA tiers")
    ax.text(1_200, ax.get_ylim()[1] * 0.95, "Low\n< 2.5 s", color=TEAL, fontsize=8, ha="center")
    ax.text(6_200, ax.get_ylim()[1] * 0.95, "Medium\n2.5 to 10 s", color="#B8860B", fontsize=8, ha="center")
    ax.text(19_500, ax.get_ylim()[1] * 0.95, "High\n> 10 s", color=RED, fontsize=8, ha="center")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "target_distribution.png")
    plt.close(fig)


def target_distribution_wide(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12.0, 2.0))
    bins = np.arange(0, 30_001, 1_000)
    ax.hist(df[TARGET].clip(upper=30_000), bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
    ax.axvspan(0, 2_500, color=TEAL, alpha=0.10)
    ax.axvspan(2_500, 10_000, color=AMBER, alpha=0.12)
    ax.axvspan(10_000, 30_000, color=RED, alpha=0.10)
    ax.set_xlim(0, 30_000)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    ax.set_xlabel("CPU build time (ms)")
    ax.set_ylabel("Builds")
    ax.set_title("Build CPU time and the three SLA tiers", fontsize=10, loc="left")
    ax.text(1_200, ax.get_ylim()[1] * 0.92, "Low < 2.5 s", color=TEAL, fontsize=8)
    ax.text(4_000, ax.get_ylim()[1] * 0.92, "Medium 2.5 to 10 s", color="#B8860B", fontsize=8)
    ax.text(12_000, ax.get_ylim()[1] * 0.92, "High > 10 s", color=RED, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "target_distribution_wide.png")
    plt.close(fig)


def analysis_signal(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for tier in ["Low", "Medium", "High"]:
        part = df[df["risk_tier"] == tier]
        ax.scatter(
            part["total_actions"].clip(lower=1),
            part[TARGET],
            s=22,
            alpha=0.72,
            color=COLORS[tier],
            label=f"{tier} risk",
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total actions in the build graph (log scale)")
    ax.set_ylabel("CPU build time, ms (log scale)")
    ax.set_title("Build graph size versus CPU time")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "analysis_signal.png")
    plt.close(fig)


def model_comparison(metrics: dict) -> None:
    rows = [
        ("Stanford baseline", metrics["baseline"]["test"]["rmse"], RED, None),
        ("Ridge", metrics["models"]["Ridge Regression"]["test"]["rmse"], BLUE, None),
        ("Random Forest", metrics["models"]["Random Forest"]["test"]["rmse"], BLUE, None),
        ("HistGradientBoosting", metrics["models"]["HistGradientBoosting"]["test"]["rmse"], TEAL, None),
        ("Post-execution reference", metrics["postexec_reference"]["test"]["rmse"], GRAY, "//"),
    ]
    rows.sort(key=lambda row: row[1], reverse=True)
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    colors = [row[2] for row in rows]
    hatches = [row[3] for row in rows]
    bars = ax.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.7)
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)
            bar.set_alpha(0.55)
    for bar, value in zip(bars, values):
        ax.text(value + 90, bar.get_y() + bar.get_height() / 2, f"{value:,.0f} ms", va="center", fontsize=8)
    ax.set_xlabel("Test RMSE (ms), lower is better")
    ax.set_title("Early-phase model versus baseline")
    ax.set_xlim(0, max(values) * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.text(
        0.5,
        -0.02,
        "Hatched bar uses actions executed, a post-execution signal, and is shown only as a reference.",
        ha="center",
        fontsize=8,
        color=GRAY,
    )
    fig.savefig(FIG_DIR / "model_comparison.png")
    plt.close(fig)


def actual_vs_predicted(pred: pd.DataFrame, metrics: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharex=True, sharey=True)
    limit = 30_000
    panels = [
        ("baseline_ms", "Stanford baseline", metrics["baseline"]["test"]),
        ("predicted_ms", "HistGradientBoosting", metrics["selected_test"]),
    ]
    for ax, (column, title, scores) in zip(axes, panels):
        ax.scatter(pred["actual_ms"], pred[column], s=22, alpha=0.72, color=BLUE, edgecolors="white", linewidths=0.3)
        ax.plot([0, limit], [0, limit], color=INK, linestyle="--", linewidth=1.2)
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("Actual CPU time (ms)")
        ax.set_title(f"{title}\nRMSE {scores['rmse']:,.0f} ms, $R^2$ {scores['r2']:.3f}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Predicted CPU time (ms)")
    fig.suptitle("Predictions on the held-out final 100 builds", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(FIG_DIR / "actual_vs_predicted.png")
    plt.close(fig)


def feature_importance(df: pd.DataFrame) -> None:
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "regressor.joblib")
    train, val, _ = split_data(df)
    x_val = preprocessor.transform(val)
    result = permutation_importance(
        model,
        x_val,
        val[TARGET].values,
        n_repeats=15,
        random_state=42,
        scoring="neg_root_mean_squared_error",
    )
    names = [_clean_feature(name) for name in preprocessor.get_feature_names_out()]
    table = (
        pd.DataFrame({"feature": names, "importance": result.importances_mean})
        .sort_values("importance", ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    ax.barh(table["feature"][::-1], table["importance"][::-1], color=TEAL, edgecolor=INK, linewidth=0.6)
    ax.set_xlabel("Increase in validation RMSE when the feature is shuffled (ms)")
    ax.set_title("What the early-phase model actually uses")
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "feature_importance.png")
    plt.close(fig)


def risk_confusion(metrics: dict) -> None:
    matrix = np.asarray(metrics["risk_classifier"]["test"]["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap=sns.light_palette(TEAL, as_cmap=True),
        cbar=False,
        xticklabels=["Low", "Medium", "High"],
        yticklabels=["Low", "Medium", "High"],
        annot_kws={"size": 13, "weight": "bold"},
        linewidths=2,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted risk tier")
    ax.set_ylabel("Actual risk tier")
    risk = metrics["risk_classifier"]["test"]
    ax.set_title(
        f"Risk classifier on the final 100 builds\n"
        f"accuracy {risk['accuracy']:.0%}, macro F1 {risk['macro_f1']:.2f}, zero High to Low",
        fontsize=10,
    )
    ax.grid(False)
    fig.savefig(FIG_DIR / "confusion_matrix_risk.png")
    plt.close(fig)


def residuals(pred: pd.DataFrame, metrics: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.hist(pred["residual_ms"], bins=28, color=BLUE, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.1)
    ax.set_xlabel("Residual, actual minus predicted (ms)")
    ax.set_ylabel("Number of builds")
    ax.set_title(
        f"Residual distribution, selected model (RMSE {metrics['selected_test']['rmse']:,.0f} ms)"
    )
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(FIG_DIR / "residuals.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = add_features(load_dataset())
    pred = pd.read_csv(PROJECT_ROOT / "reports" / "test_predictions.csv")
    metrics = json.loads((PROJECT_ROOT / "reports" / "metrics.json").read_text())

    target_distribution(df)
    target_distribution_wide(df)
    analysis_signal(df)
    model_comparison(metrics)
    actual_vs_predicted(pred, metrics)
    feature_importance(df)
    risk_confusion(metrics)
    residuals(pred, metrics)
    print("Saved figures to", FIG_DIR)
    for path in sorted(FIG_DIR.glob("*.png")):
        print(" ", path.name)


if __name__ == "__main__":
    main()
