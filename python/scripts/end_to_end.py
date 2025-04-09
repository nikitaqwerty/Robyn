#!/usr/bin/env python3
"""
Marketing Mix Modeling (MMM) Script using Robyn
This script runs the complete Robyn MMM workflow with all hyperparameters defined at the top.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import base64
import plotly.io as pio
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy
from IPython.display import Image, display
import shutil

# Robyn imports
from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import AdstockType, DependentVarType
from robyn.modeling.feature_engineering import FeatureEngineering
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.model_executor import ModelExecutor
from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
from robyn.modeling.clustering.clustering_config import ClusteringConfig, ClusterBy
from robyn.modeling.clustering.cluster_builder import ClusterBuilder
from robyn.modeling.pareto.pareto_utils import ParetoUtils
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from robyn.visualization.pareto_visualizer import ParetoVisualizer
from robyn.allocator.entities.allocation_params import AllocatorParams
from robyn.allocator.constants import SCENARIO_MAX_RESPONSE, CONSTRAINT_MODE_EQ
from robyn.allocator.optimizer import BudgetAllocator
from robyn.visualization.allocator_visualizer import AllocatorPlotter

# ============== HYPERPARAMETERS AND CONFIGURATION ==============

# Base path that will be prepended to DATA_PATH and WORKING_DIR
BASE_PATH = Path("../mmm")

# Basic configuration
VERSION = "0.5.test"
DATA_PATH = BASE_PATH / f'data/{".".join(VERSION.split(".")[0:2])}'
WORKING_DIR = BASE_PATH / f"output/robyn_output_{VERSION}"
START_DATE = "2022-05-01"
END_DATE = "2025-01-01"

# Channel-specific hyperparameters
CHANNEL_HYPERPARAMETERS = {
    "AffiliatesCPA_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0, 0.3],
    },
    "Comms_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0.2, 0.9],
    },
    "Digital_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0, 0.4],
    },
    "Mobile_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0, 0.3],
    },
    "Sponsorship_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0.3, 0.9],
    },
    "Offline_spend": {
        "alphas": [0.5, 3],
        "gammas": [0.3, 1],
        "thetas": [0.3, 0.9],
    },
}

# General model hyperparameters
ADSTOCK_TYPE = AdstockType.GEOMETRIC
LAMBDA_RANGE = [0, 1]
TRAIN_SIZE_RANGE = [0.8, 1.0]

# Model execution parameters
TRIALS_CONFIG = {"iterations": 1000, "trials": 2}

MODEL_EXECUTION_PARAMS = {
    "ts_validation": True,
    "add_penalty_factor": False,
    "rssd_zero_penalty": True,
    "nevergrad_algo": NevergradAlgorithm.TWO_POINTS_DE,
    "objective_weights": [1, 1],
    "model_name": Models.RIDGE,
    "cores": 16,
    "seed": [42],
    "val_size": 5,
    "test_size": 5,
    "fixed_coefficients": {"AffiliatesCPA_spend": 7000.0},
    "fixed_intercept": 843.66,
}

# Clustering parameters
CLUSTERING_CONFIG = {
    "max_clusters": 10,
    "min_clusters": 3,
    "weights": [1.0, 1.0, 1.0],
}

# Budget allocation parameters
BUDGET_ALLOCATION_PARAMS = {
    "scenario": SCENARIO_MAX_RESPONSE,
    "total_budget": None,  # Uses total spend in date_range when None
    "date_range": "last",
    "channel_constr_low": [0.75, 0.75, 0.75, 0.75, 1, 1],  # Minimum spend multiplier
    "channel_constr_up": [1.25, 1.25, 1.25, 1.25, 1, 1],  # Maximum spend multiplier
    "channel_constr_multiplier": 1.5,
    "optim_algo": "SLSQP_AUGLAG",
    "maxeval": 100000,
    "constr_mode": CONSTRAINT_MODE_EQ,
    "plots": True,
}


def clear_output_directory(directory_path):
    """Clear the contents of the output directory if it exists"""
    path = Path(directory_path)
    if path.exists():
        print(f"Clearing existing output directory: {path}")
        shutil.rmtree(path)

    # Create fresh directory
    path.mkdir(parents=True, exist_ok=True)
    print(f"Created fresh output directory: {path}")


def main():
    """Main function executing the entire MMM workflow"""
    # Clear output directory before running
    clear_output_directory(WORKING_DIR)

    # Create data directory if needed
    DATA_PATH.mkdir(parents=True, exist_ok=True)

    # Set up plotly renderer
    pio.renderers.default = "iframe"

    print(f"Starting MMM analysis with Robyn {VERSION}")

    # Load data
    df = pd.read_csv(DATA_PATH.joinpath("weekly_data.csv"))

    # Drop unnecessary columns
    df = df.drop(columns={"pari_nn_event", "blast_event"})

    # Display data summary
    print("Data tail:")
    print(df.tail(10).round(0))

    # Rename target variable and convert date
    df = df.rename(columns={"FTD": "dep_var"})
    df.Date = pd.to_datetime(df.Date)

    # Identify spend and event columns
    spend_cols = [col for col in df.columns if col.endswith("_spend")]
    event_cols = [col for col in df.columns if col.endswith("_event")]
    print("spend cols", spend_cols)
    print("event_cols", event_cols)

    # Check last week spend
    print(f"Last week total spend: {df.iloc[-1][spend_cols].sum()}")

    # Create MMMData specification
    mmm_data_spec = MMMData.MMMDataSpec(
        dep_var="dep_var",  # Target variable
        dep_var_type="conversion",  # Type: "revenue" or "conversion"
        date_var="Date",  # Date column name
        context_vars=event_cols + ["MeanBonusAmount"],  # External factors
        paid_media_spends=spend_cols,  # Media spend columns
        paid_media_vars=[],  # Media metrics
        organic_vars=[],  # Non-paid marketing activities
        window_start=START_DATE,  # Analysis start date
        window_end=END_DATE,  # Analysis end date
    )

    # Create MMMData object
    mmm_data = MMMData(data=df, mmmdata_spec=mmm_data_spec)

    # Load holidays data
    holidays_file = BASE_PATH / "data/dt_prophet_holidays.csv"
    dt_prophet_holidays = pd.read_csv(holidays_file)
    holidays_data = HolidaysData(
        dt_holidays=dt_prophet_holidays,
        prophet_vars=["trend", "season", "holiday"],
        prophet_country="RU",
        prophet_signs=["default", "default", "default"],
    )

    # Set up hyperparameters for each channel
    channel_hyperparams = {}
    for channel, params in CHANNEL_HYPERPARAMETERS.items():
        channel_hyperparams[channel] = ChannelHyperparameters(
            alphas=params["alphas"],
            gammas=params["gammas"],
            thetas=params["thetas"],
        )

    # Create hyperparameters object
    hyperparameters = Hyperparameters(
        hyperparameters=channel_hyperparams,
        adstock=ADSTOCK_TYPE,
        lambda_=LAMBDA_RANGE,
        train_size=TRAIN_SIZE_RANGE,
    )

    # Initialize Robyn instance
    robyn = Robyn(working_dir=str(WORKING_DIR))

    # Feature engineering
    print("Performing feature engineering...")
    feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
    featurized_mmm_data = feature_engineering.perform_feature_engineering()

    # Set up trials configuration
    trials_config = TrialsConfig(
        iterations=TRIALS_CONFIG["iterations"], trials=TRIALS_CONFIG["trials"]
    )

    # Create model executor
    print("Setting up model executor...")
    model_executor = ModelExecutor(
        mmmdata=mmm_data,
        holidays_data=holidays_data,
        hyperparameters=hyperparameters,
        calibration_input=None,
        featurized_mmm_data=featurized_mmm_data,
    )

    # Run the model
    print(
        f"Running model with {TRIALS_CONFIG['iterations']} iterations and {TRIALS_CONFIG['trials']} trials..."
    )
    output_models = model_executor.model_run(
        trials_config=trials_config,
        ts_validation=MODEL_EXECUTION_PARAMS["ts_validation"],
        add_penalty_factor=MODEL_EXECUTION_PARAMS["add_penalty_factor"],
        rssd_zero_penalty=MODEL_EXECUTION_PARAMS["rssd_zero_penalty"],
        nevergrad_algo=MODEL_EXECUTION_PARAMS["nevergrad_algo"],
        objective_weights=MODEL_EXECUTION_PARAMS["objective_weights"],
        model_name=MODEL_EXECUTION_PARAMS["model_name"],
        cores=MODEL_EXECUTION_PARAMS["cores"],
        seed=MODEL_EXECUTION_PARAMS["seed"],
        val_size=MODEL_EXECUTION_PARAMS["val_size"],
        test_size=MODEL_EXECUTION_PARAMS["test_size"],
        fixed_coefficients=MODEL_EXECUTION_PARAMS["fixed_coefficients"],
        fixed_intercept=MODEL_EXECUTION_PARAMS["fixed_intercept"],
    )

    # Display model output summaries
    print("Model outputs summary:")
    print(output_models.all_x_decomp_agg.head())
    print(output_models.all_result_hyp_param.head())
    print(
        "Best MAE coeffs:",
        output_models.all_x_decomp_agg.loc[
            output_models.all_x_decomp_agg.mae
            == output_models.all_x_decomp_agg.mae.min()
        ],
    )

    # Display convergence plots if available
    if "moo_cloud_plot" in output_models.convergence:
        moo_cloud_plot = output_models.convergence["moo_cloud_plot"]
        display(Image(data=base64.b64decode(moo_cloud_plot)))

    if "moo_distrb_plot" in output_models.convergence:
        moo_distrb_plot = output_models.convergence["moo_distrb_plot"]
        display(Image(data=base64.b64decode(moo_distrb_plot)))

    # Run Pareto optimization
    print("Running Pareto optimization...")
    pareto_optimizer = ParetoOptimizer(
        mmm_data, output_models, hyperparameters, featurized_mmm_data, holidays_data
    )

    pareto_result = pareto_optimizer.optimize(pareto_fronts="auto", min_candidates=100)
    unfiltered_pareto_result = copy.deepcopy(pareto_result)

    print("Unfiltered Pareto results:")
    print(
        unfiltered_pareto_result.result_hyp_param.sort_values(
            "mae", ascending=True
        ).head()
    )

    # Run clustering
    print("Running clustering...")
    cluster_configs = ClusteringConfig(
        dep_var_type=DependentVarType(mmm_data.mmmdata_spec.dep_var_type),
        cluster_by=ClusterBy.HYPERPARAMETERS,
        max_clusters=CLUSTERING_CONFIG["max_clusters"],
        min_clusters=CLUSTERING_CONFIG["min_clusters"],
        weights=CLUSTERING_CONFIG["weights"],
    )

    cluster_builder = ClusterBuilder(pareto_result=pareto_result)
    cluster_results = cluster_builder.cluster_models(cluster_configs)

    # Process Pareto clustered results
    print("Processing Pareto clustered results...")
    utils = ParetoUtils()
    pareto_result = utils.process_pareto_clustered_results(
        pareto_result,
        clustered_result=cluster_results,
        ran_cluster=True,
        ran_calibration=False,
    )

    print("Filtered Pareto results:")
    print(pareto_result.result_hyp_param.sort_values("mae", ascending=True).head())

    # Analyze best model
    best_mae_row = pareto_result.result_hyp_param.sort_values(
        "mae", ascending=True
    ).iloc[0]
    best_mae_model = best_mae_row.sol_id
    print(f"Best model: {best_mae_model}, mae: {best_mae_row.mae}")

    plot_data = pareto_result.plot_data_collect[best_mae_model]
    ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"]
    print("Time series data:")
    print(ts_data.tail(10))

    # Analyze time series data
    def reshape_and_analyze_ts_data(df):
        """Reshape melted time series data and calculate metrics for the last 10 rows."""
        # Reshape from long to wide format
        wide_df = df.pivot(index="ds", columns="variable", values="value").reset_index()

        # Ensure column names are clean
        wide_df.columns.name = None

        # Sort by date ascending
        wide_df = wide_df.sort_values("ds")

        # Get the last 10 rows
        last_10 = wide_df.tail(10)

        # Calculate metrics on the last 10 rows
        metrics = {}

        # Mean Absolute Error
        metrics["mae"] = mean_absolute_error(last_10["actual"], last_10["predicted"])

        # Root Mean Square Error
        metrics["rmse"] = np.sqrt(
            mean_squared_error(last_10["actual"], last_10["predicted"])
        )

        # R-squared
        metrics["r2"] = r2_score(last_10["actual"], last_10["predicted"])

        return wide_df, last_10, metrics

    wide_data, last_10_rows, metrics = reshape_and_analyze_ts_data(ts_data)

    # Display the results for original data
    print("Original Data - Metrics for the last 10 data points:")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    print(f"R2: {metrics['r2']:.4f}")

    print("\nOriginal data (last 10 rows):")
    print(last_10_rows)

    # Visualize Pareto results
    print("Visualizing Pareto results...")
    pareto_visualizer = ParetoVisualizer(
        pareto_result=pareto_result,
        mmm_data=mmm_data,
        holiday_data=holidays_data,
        hyperparameter=hyperparameters,
        featurized_mmm_data=featurized_mmm_data,
        unfiltered_pareto_result=unfiltered_pareto_result,
        model_outputs=output_models,
    )
    pareto_visualizer.plot_all(
        display_plots=True,
        export_location=str(WORKING_DIR),
        display_criteria="best_mae_test",
    )

    # Visualize cluster results
    print("Visualizing cluster results...")
    cluster_visualizer = ClusterVisualizer(
        pareto_result,
        cluster_results,
        mmm_data,
    )
    cluster_visualizer.plot_all(display_plots=True, export_location=str(WORKING_DIR))

    # Select model for budget allocation
    select_model = (
        pareto_result.result_hyp_param.sort_values("mae", ascending=False)
        .iloc[0]
        .sol_id
    )
    print(f"Selected model for budget allocation: {select_model}")

    # Run budget allocation
    print("Running budget allocation...")
    allocator_params = AllocatorParams(
        scenario=BUDGET_ALLOCATION_PARAMS["scenario"],
        total_budget=BUDGET_ALLOCATION_PARAMS["total_budget"],
        date_range=BUDGET_ALLOCATION_PARAMS["date_range"],
        channel_constr_low=BUDGET_ALLOCATION_PARAMS["channel_constr_low"],
        channel_constr_up=BUDGET_ALLOCATION_PARAMS["channel_constr_up"],
        channel_constr_multiplier=BUDGET_ALLOCATION_PARAMS["channel_constr_multiplier"],
        optim_algo=BUDGET_ALLOCATION_PARAMS["optim_algo"],
        maxeval=BUDGET_ALLOCATION_PARAMS["maxeval"],
        constr_mode=BUDGET_ALLOCATION_PARAMS["constr_mode"],
        plots=BUDGET_ALLOCATION_PARAMS["plots"],
    )

    # Initialize budget allocator
    max_response_allocator = BudgetAllocator(
        mmm_data=mmm_data,
        featurized_mmm_data=featurized_mmm_data,
        hyperparameters=hyperparameters,
        pareto_result=pareto_result,
        select_model=select_model,
        params=allocator_params,
    )

    # Run optimization
    max_response_result = max_response_allocator.optimize()

    # Display allocation results
    results_df = pd.DataFrame(
        {
            "Channel": max_response_result.dt_optimOut.channels,
            "Initial Spend": max_response_result.dt_optimOut.init_spend_unit,
            "Optimized Spend": max_response_result.dt_optimOut.optm_spend_unit,
            "Spend Change %": (
                max_response_result.dt_optimOut.optm_spend_unit
                / max_response_result.dt_optimOut.init_spend_unit
                - 1
            )
            * 100,
            "Initial Response": max_response_result.dt_optimOut.init_response_unit,
            "Optimized Response": max_response_result.dt_optimOut.optm_response_unit,
            "Response Lift %": (
                max_response_result.dt_optimOut.optm_response_unit
                / max_response_result.dt_optimOut.init_response_unit
                - 1
            )
            * 100,
        }
    )

    print("Budget allocation results:")
    print(results_df.round(2))

    # Create allocation plots
    print("Creating allocation visualizations...")
    plotter = AllocatorPlotter(
        allocation_result=max_response_result, budget_allocator=max_response_allocator
    )

    plots = plotter.plot_all(display_plots=True, export_location=str(WORKING_DIR))

    print(f"MMM analysis complete. All outputs saved to {WORKING_DIR}")


if __name__ == "__main__":
    main()
