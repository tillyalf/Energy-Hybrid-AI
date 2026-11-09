"""
Student Name: Matilda Alford
Student FAN: ALFO0043
File: evaluate_renewable.py
Description: Evaluate renewable forecasting on chronologically held-out data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from threadpoolctl import threadpool_limits

from models.RenewableModel import RenewableModel


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data/processed/electricity_cleaned.parquet"


def main():
    # Load cleaned historical data.
    history = (
        pd.read_parquet(DATA_PATH)
        .set_index("timestamp")
        .sort_index()
    )

    # Reuse the exact feature engineering from RenewableModel.
    features = RenewableModel.build_features(history)

    # Predict renewable generation one hour ahead.
    target = history["generation_renewable"].shift(
        -RenewableModel.HORIZON_HOURS
    )

    # Remove rows with missing predictors or targets.
    examples = features.copy()
    examples["target"] = target
    examples["persistence"] = history["generation_renewable"]
    examples = examples.replace([np.inf, -np.inf], np.nan).dropna()

    # Chronological 80/20 split.
    split = int(len(examples) * 0.8)
    train = examples.iloc[:split]
    test = examples.iloc[split:]

    # Same hyperparameters as the deployment model.
    model = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=42,
    )

    with threadpool_limits(limits=2):
        model.fit(train[RenewableModel.FEATURES], train["target"])
        predictions = model.predict(test[RenewableModel.FEATURES])

    actual = test["target"]
    baseline = test["persistence"]

    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    r2 = r2_score(actual, predictions)
    baseline_mae = mean_absolute_error(actual, baseline)

    skill = 1 - mae / baseline_mae

    print("\n========== Renewable Forecast Evaluation =========")
    print(f"Training examples: {len(train):,}")
    print(f"Testing examples: {len(test):,}")
    print(f"Test period: {test.index.min()} to {test.index.max()}")
    print(f"MAE: {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"R²: {r2:.4f}")
    print(f"Persistence MAE: {baseline_mae:.2f} MW")
    print(f"Skill vs persistence: {skill:.2%}")


if __name__ == "__main__":
    main()
