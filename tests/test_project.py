"""Small smoke tests for the data and model artifacts.

Run with:
    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import math
import sys
import unittest
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_dataset  # noqa: E402
from features import ANALYSIS_NUM_COLS, FEATURE_COLS, RISK_LABELS, split_data  # noqa: E402
from predict import predict_build  # noqa: E402


class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_dataset()
        cls.metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())

    def test_dataset_shape(self) -> None:
        self.assertEqual(len(self.data), 500)
        required = {
            "cpu_time_ms",
            "packages_loaded",
            "targets_configured",
            "total_actions",
            "risk_tier",
        }
        self.assertTrue(required.issubset(self.data.columns))
        self.assertEqual(int(self.data[list(required)].isna().sum().sum()), 0)

    def test_split_sizes(self) -> None:
        train, validation, test = split_data(self.data)
        self.assertEqual((len(train), len(validation), len(test)), (300, 100, 100))

    def test_no_post_execution_feature(self) -> None:
        self.assertNotIn("actions_executed", FEATURE_COLS)
        self.assertNotIn("actions_executed", ANALYSIS_NUM_COLS)
        self.assertNotIn("cache_hit_ratio", FEATURE_COLS)

    def test_saved_models_load(self) -> None:
        for name in ["preprocessor.joblib", "regressor.joblib", "risk_classifier.joblib"]:
            self.assertTrue((ROOT / "models" / name).exists(), name)
        risk = joblib.load(ROOT / "models" / "risk_classifier.joblib")
        self.assertEqual(sorted(risk.classes_.tolist()), sorted(RISK_LABELS))

    def test_cli_prediction(self) -> None:
        result = predict_build("src/main", "JAVA", 4, 1462, 2883)
        self.assertTrue(math.isfinite(result["predicted_ms"]))
        self.assertGreater(result["predicted_ms"], 0)
        self.assertIn(result["risk_tier"], RISK_LABELS)

    def test_reported_metrics(self) -> None:
        self.assertEqual(self.metrics["selected_model"], "HistGradientBoosting")
        self.assertAlmostEqual(self.metrics["selected_test"]["rmse"], 3158.65, places=1)
        self.assertAlmostEqual(self.metrics["selected_test"]["r2"], 0.702, places=2)
        self.assertEqual(self.metrics["risk_classifier"]["test"]["high_to_low"], 0)


if __name__ == "__main__":
    unittest.main()
