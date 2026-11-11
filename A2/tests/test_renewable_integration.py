"""
Student Name: Matilda Alford
Student FAN: ALFO0043
File: test_renewable_integration.py
Description: Integration tests for renewable forecasting and symbolic inference.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import run_inference


class RenewableInferenceTests(unittest.TestCase):

    def setUp(self):
        """Create shared sample forecasts for integration testing."""
        self.time = pd.Timestamp("2026-11-03T14:00:00+10:00")

        self.history = pd.DataFrame(
            {"generation_renewable": [2300.0], "demand": [4900.0]},
            index=pd.DatetimeIndex(
                [self.time - pd.Timedelta(hours=1)],
                name="timestamp",
            ),
        )

        self.renewable = {
            "forecast_time": self.time,
            "generation_renewable_mw": 2400.0,
            "horizon_hours": 1,
        }

        self.demand = {
            "forecast_time": self.time,
            "demand_mw": 5000.0,
            "horizon_hours": 1,
            "lower_95_mw": 4000.0,
            "upper_95_mw": 6000.0,
        }

    def run_mocked_inference(self, renewable=None, demand=None):
        """Run actual renewable rules with mocked ML subsystems."""

        with (
            patch("inference.RenewableModel") as renewable_model,
            patch("inference.DemandModel") as demand_model,
            patch("inference.NetworkStressModel") as network_model,
            patch("inference.PriceModel") as price_model,
            patch("inference.SystemAssessment") as system_assessment,
            patch("inference.print_assessment"),
        ):
            renewable_model.return_value.inference.return_value = (
                renewable if renewable is not None else self.renewable
            )

            if demand is None:
                demand_model.return_value.inference.side_effect = (
                    ValueError("Demand forecast unavailable")
                )
            else:
                demand_model.return_value.inference.return_value = demand

            # Other subsystems are not required for renewable coverage tests.
            network_model.return_value.inference.side_effect = (
                ValueError("Network test skipped")
            )
            price_model.return_value.inference.side_effect = (
                ValueError("Price test skipped")
            )

            system_assessment.return_value.assess.return_value = {
                "priority": "manual_review_required"
            }

            results = run_inference(self.history)

            return results, renewable_model, demand_model

    def test_renewable_target_missed(self):
        """2400 MW renewable against 5000 MW demand misses 50% target."""

        results, renewable_model, _ = self.run_mocked_inference(
            demand=self.demand
        )

        assessment = results["subsystems"]["coverage"]

        self.assertEqual(assessment["status"], "insufficient")
        self.assertEqual(assessment["renewable_shortfall_mw"], 100.0)
        self.assertAlmostEqual(assessment["coverage_percent"], 48.0)
        self.assertIn("target_missed", assessment["rule_ids"])

        pd.testing.assert_frame_equal(
            renewable_model.return_value.inference.call_args.args[0],
            self.history,
        )

    def test_missing_demand_prevents_coverage(self):
        """Missing demand must not produce a sufficiency classification."""

        results, _, demand_model = self.run_mocked_inference(
            demand=None
        )

        self.assertIn("renewable", results["subsystems"])
        self.assertNotIn("coverage", results["subsystems"])
        self.assertIn("coverage", results["errors"])
        self.assertIn("demand", results["errors"])

        demand_model.return_value.inference.assert_called_once()

    def test_demand_forecast_drives_sufficiency(self):
        """Demand prediction must determine renewable coverage."""

        results, renewable_model, demand_model = (
            self.run_mocked_inference(demand=self.demand)
        )

        self.assertEqual(results["subsystems"]["demand"], self.demand)

        assessment = results["subsystems"]["coverage"]

        self.assertEqual(assessment["status"], "insufficient")
        self.assertEqual(assessment["renewable_shortfall_mw"], 100.0)
        self.assertEqual(
            assessment["demand_sensitivity"],
            "depends_on_demand",
        )

        pd.testing.assert_frame_equal(
            demand_model.return_value.inference.call_args.args[0],
            renewable_model.return_value.inference.call_args.args[0],
        )

        self.assertFalse(
            demand_model.return_value.inference.call_args.kwargs["verbose"]
        )

    def test_mismatched_forecast_timestamps(self):
        """Different forecast hours must not produce renewable coverage."""

        invalid_demand = {
            **self.demand,
            "forecast_time": self.time + pd.Timedelta(hours=1),
        }

        results, _, _ = self.run_mocked_inference(
            demand=invalid_demand
        )

        self.assertNotIn("coverage", results["subsystems"])
        self.assertIn("coverage", results["errors"])
        self.assertIn(
            "timestamps must match",
            results["errors"]["coverage"],
        )

    def test_demand_returns_prediction_and_preserves_default_explanation(self):
        """Verify demand inference can optionally skip explanations."""

        from models.DemandModel import DemandModel

        index = pd.date_range(
            "2026-01-01",
            periods=200,
            freq="h",
            tz="Etc/GMT-10",
        )

        history = pd.DataFrame(
            {
                "demand": 5000.0,
                "demand_lag_1h": 5000.0,
                "is_weekend": 0,
            },
            index=index,
        )

        archive = {
            "theta": np.zeros(1),
            "X_train": np.zeros((10, 7)),
            "y_train": np.zeros(10),
            "hour_scale": np.ones(24),
        }

        gp = MagicMock()
        gp.fit.return_value = gp
        gp.predict.return_value = (
            np.array([10.0]),
            np.array([20.0]),
        )

        model = DemandModel()

        with (
            patch("models.DemandModel.np.load", return_value=archive),
            patch.object(model, "_make_kernel", return_value=MagicMock()),
            patch.object(model, "_gp", return_value=gp),
            patch.object(model, "_explain") as explain,
        ):
            forecast = model.inference(history, explain=False)

            explain.assert_not_called()

            self.assertEqual(
                forecast["forecast_time"],
                index[-1] + pd.Timedelta(hours=1),
            )

            self.assertEqual(forecast["demand_mw"], 5010.0)

            self.assertAlmostEqual(
                forecast["upper_95_mw"] - forecast["demand_mw"],
                39.2,
            )

            model.inference(history)
            explain.assert_called_once()


if __name__ == "__main__":
    unittest.main()
