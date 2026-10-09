"""Load the Stanford Bazel dataset and extract analysis-phase build signals.

The raw dataset has three parts:
  * InputData.csv : commit id, dominant changed directory, dominant file type
  * CPUTimes.csv  : CPU time for each build, in milliseconds
  * Build_Info/   : one Build Event Protocol (BEP) JSON stream per build

Only signals that are available before compilation actions start are used by
the model.  Those are the final "packages loaded / targets configured" counts
printed at the end of Bazel's analysis phase, and the total number of actions
in the action graph.  The number of actions that actually ran is kept as a
reference column, but it is never used as a model input.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_CSV = PROCESSED_DIR / "bazel_build_dataset.csv"

LOW_RISK_MS = 2_500
HIGH_RISK_MS = 10_000

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ANALYZED = re.compile(
    r"INFO: Analyzed target[^\n]*?\(([\d,]+) packages loaded, ([\d,]+) targets configured\)"
)
_ANALYZING = re.compile(
    r"Analyzing: target[^\n]*?\(([\d,]+) packages loaded, ([\d,]+) targets configured\)"
)
_PROGRESS = re.compile(r"\[([\d,]+)\s*/\s*([\d,]+)\]")
_TOTAL = re.compile(r"INFO: Build completed successfully,\s*([\d,]+) total actions")


def _as_int(value: str) -> int:
    return int(value.replace(",", ""))


def parse_bep_file(path: Path) -> dict:
    """Return analysis-phase and reference metrics from one BEP JSONL file."""
    analyzed: list[tuple[int, int]] = []
    analyzing: list[tuple[int, int]] = []
    denominators: list[int] = []
    final_totals: list[int] = []

    record = {
        "packages_loaded": np.nan,
        "targets_configured": np.nan,
        "total_actions": np.nan,
        "actions_executed": np.nan,
        "wall_time_ms": np.nan,
        "cpu_time_bep_ms": np.nan,
        "success": np.nan,
    }

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if "progress" in event:
                text = (event["progress"].get("stderr") or "") + " " + (
                    event["progress"].get("stdout") or ""
                )
                if text:
                    text = _ANSI.sub("", text).replace("\r", "\n")
                    analyzed.extend(
                        (_as_int(a), _as_int(b)) for a, b in _ANALYZED.findall(text)
                    )
                    analyzing.extend(
                        (_as_int(a), _as_int(b)) for a, b in _ANALYZING.findall(text)
                    )
                    denominators.extend(_as_int(b) for _, b in _PROGRESS.findall(text))
                    final_totals.extend(_as_int(a) for a in _TOTAL.findall(text))

            if "buildMetrics" in event:
                metrics = event["buildMetrics"]
                actions = metrics.get("actionSummary", {})
                timing = metrics.get("timingMetrics", {})
                if "actionsExecuted" in actions:
                    record["actions_executed"] = int(actions["actionsExecuted"])
                if "wallTimeInMs" in timing:
                    record["wall_time_ms"] = int(timing["wallTimeInMs"])
                if "cpuTimeInMs" in timing:
                    record["cpu_time_bep_ms"] = int(timing["cpuTimeInMs"])

            if "finished" in event and "overallSuccess" in event["finished"]:
                record["success"] = bool(event["finished"]["overallSuccess"])

    if analyzed:
        record["packages_loaded"], record["targets_configured"] = analyzed[-1]
    elif analyzing:
        record["packages_loaded"], record["targets_configured"] = max(
            analyzing, key=lambda pair: pair[1]
        )

    if denominators:
        record["total_actions"] = max(denominators)
    elif final_totals:
        record["total_actions"] = max(final_totals)

    return record


def _risk_tier(cpu_ms: float) -> str:
    if cpu_ms < LOW_RISK_MS:
        return "Low"
    if cpu_ms <= HIGH_RISK_MS:
        return "Medium"
    return "High"


def build_dataset(raw_dir: Path = RAW_DIR, out_path: Path = PROCESSED_CSV) -> pd.DataFrame:
    """Rebuild the processed dataset from the raw files and save it as CSV."""
    input_df = pd.read_csv(raw_dir / "InputData.csv")
    cpu_df = pd.read_csv(raw_dir / "CPUTimes.csv")

    if len(input_df) != 500:
        raise ValueError(f"Expected 500 commits, found {len(input_df)}")
    if not cpu_df["commitID"].tolist() == list(range(1, 501)):
        raise ValueError("CPUTimes.csv is not ordered from commit 1 to 500")

    input_df["ID"] = input_df["ID"].astype(str).str.strip()
    input_df["prefix"] = input_df["prefix"].astype(str).str.strip()
    input_df["type"] = input_df["type"].astype(str).str.strip()

    build_dir = raw_dir / "Build_Info"
    rows = []
    for commit_num in range(1, 501):
        path = build_dir / f"{commit_num}_commit.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing BEP file: {path}")
        row = {"commit_num": commit_num}
        row.update(parse_bep_file(path))
        rows.append(row)

    bep_df = pd.DataFrame(rows)
    df = pd.DataFrame(
        {
            "commit_num": np.arange(1, 501),
            "commit_id": input_df["ID"].values,
            "prefix": input_df["prefix"].values,
            "type": input_df["type"].values,
            "cpu_time_ms": cpu_df["CPUTime"].astype(float).values,
        }
    ).merge(bep_df, on="commit_num", how="left")

    df["prefix_x_type"] = df["prefix"] + "_x_" + df["type"]
    df["is_build_file"] = (df["type"] == "Starlark").astype(int)
    df["is_core_src"] = (df["prefix"] == "src/main").astype(int)
    df["is_test"] = (df["prefix"] == "src/test").astype(int)
    df["risk_tier"] = df["cpu_time_ms"].map(_risk_tier)

    required = ["packages_loaded", "targets_configured", "total_actions", "cpu_time_ms"]
    if df[required].isna().any().any():
        missing = df[required].isna().sum().to_dict()
        raise ValueError(f"Missing analysis-phase values: {missing}")

    # The BEP CPU time should agree with CPUTimes.csv.  If it does not, the
    # source files are misaligned and every downstream number is suspect.
    both = df.dropna(subset=["cpu_time_bep_ms"])
    if len(both) and (both["cpu_time_bep_ms"] != both["cpu_time_ms"]).any():
        mismatches = int((both["cpu_time_bep_ms"] != both["cpu_time_ms"]).sum())
        raise ValueError(f"{mismatches} CPU time mismatches between BEP and CPUTimes.csv")

    df = df[
        [
            "commit_num",
            "commit_id",
            "prefix",
            "type",
            "cpu_time_ms",
            "packages_loaded",
            "targets_configured",
            "total_actions",
            "actions_executed",
            "wall_time_ms",
            "success",
            "prefix_x_type",
            "is_build_file",
            "is_core_src",
            "is_test",
            "risk_tier",
        ]
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_dataset(path: Path = PROCESSED_CSV) -> pd.DataFrame:
    if not path.exists():
        return build_dataset(out_path=path)
    return pd.read_csv(path)


if __name__ == "__main__":
    data = build_dataset()
    print(f"Saved {len(data)} builds to {PROCESSED_CSV}")
    print(data[["cpu_time_ms", "packages_loaded", "targets_configured", "total_actions"]].describe())
    print("\nRisk tiers:")
    print(data["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]))
