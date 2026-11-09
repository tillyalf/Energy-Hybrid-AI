"""
Student Names: Oliver Wuttke, Hans Pujalte, Shivansh Pant, Matilda Alford
Student FANs: WUTT0019, PUJA0009, PANT0108, ALFO0043
Date: 02-11-2026
File: training_pipeline.py
Description: Train all implemented A2 models on the transformed electricity data.
Usage: python training_pipeline --input data/processed/electricity_transformed.parquet
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from models.DemandModel import DemandModel
from models.NetworkStressModel import NetworkStressModel
from models.PriceModel import PriceModel
from models.RenewableModel import RenewableModel


def main():
    """Load the shared dataset and train each implemented model."""
    # CLI arg parser for inputting training set path.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to training dataset")
    args = parser.parse_args()

    # Require a dataset path before running either training pipeline.
    if args.input is not None:
        TRAINING_PATH = args.input
    else:
        sys.exit(
            "No dataset argument found. Please provide "
            "--input A2/data/processed/electricity_transformed.parquet "
            "from the repository root, as per the README.md."
        )

    # Load and sort the dataset once for both models.
    df = pd.read_parquet(TRAINING_PATH).set_index("timestamp").sort_index()

    # Train and evaluate demand prediction, then save its parameters.
    print("\n========== Demand Forecasting Model =========")
    demand_model = DemandModel()
    demand_model.train_eval(training_set=df)

    # Price-spike Prediction Model
    print('\n========== Price Spike Prediction Model =========')
    price_model = PriceModel()
    price_model.train_eval(training_set=df, run_hybrid_eval=False)

    print('\n========== Network Stress Prediction Model =========')
    # Compare network-stress classifiers and save the selected model.
    network_stress_model = NetworkStressModel()
    network_stress_model.train_eval(training_set=df)

    print('\n========== Renewable Generation Forecasting Model =========')
    # Train renewable forecasting model using cleaned data.
    # Retain historical observations excluded by curtailment trimming.
    cleaned_path = Path(TRAINING_PATH).with_name("electricity_cleaned.parquet")
    renewable_df = (
        pd.read_parquet(cleaned_path)
        .set_index("timestamp")
        .sort_index()
    )
    renewable_model = RenewableModel()
    renewable_model.train(training_set=renewable_df)

if __name__ == "__main__":
    main()

