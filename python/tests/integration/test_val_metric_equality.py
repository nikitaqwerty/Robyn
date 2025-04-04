import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("robyn_mmm")

# Configuration
version = "0.5.1tst"
logger.info(f"Starting MMM analysis with version {version}")

data_path = Path(f"/Users/nrotanov/pari/mmm/data/{'.'.join(version.split('.')[0:2])}")
data_path.mkdir(parents=True, exist_ok=True)
logger.info(f"Data path: {data_path}")

WORKING_DIR = f"/Users/nrotanov/pari/mmm/output/robyn_output_{version}"
logger.info(f"Working directory: {WORKING_DIR}")

START_DATE = "2022-05-01"
END_DATE = "2025-01-01"
logger.info(f"Analysis window: {START_DATE} to {END_DATE}")

# Import Robyn components
from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import AdstockType, DependentVarType

# Load data
logger.info("Loading data from CSV...")
df = pd.read_csv(data_path.joinpath("weekly_data.csv"))
logger.info(f"Loaded data with shape: {df.shape}")

# Data preparation
logger.info("Preparing data for analysis...")
df = df.rename(columns={"FTD": "dep_var"})
df.Date = pd.to_datetime(df.Date)
logger.info("Renamed 'FTD' to 'dep_var' and converted Date to datetime")

# Identify column types
spend_cols = [col for col in df.columns if col.endswith("_spend")]
event_cols = [col for col in df.columns if col.endswith("_event")]
logger.info(f"Identified {len(spend_cols)} spend columns: {spend_cols}")
logger.info(f"Identified {len(event_cols)} event columns: {event_cols}")

# Create MMM data specification
logger.info("Creating MMM data specification...")
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

mmm_data = MMMData(data=df, mmmdata_spec=mmm_data_spec)
logger.info("MMM data specification created successfully")

# Setup holidays data
logger.info("Setting up holidays data...")
dt_prophet_holidays = pd.read_csv(f"{data_path.parent}/dt_prophet_holidays.csv")
logger.info(f"Loaded holidays data with {len(dt_prophet_holidays)} rows")

holidays_data = HolidaysData(
    dt_holidays=dt_prophet_holidays,
    prophet_vars=["trend", "season", "holiday"],
    prophet_country="RU",
    prophet_signs=["default", "default", "default"],
)
logger.info("Holidays data setup completed")

# Setup hyperparameters
logger.info("Setting up model hyperparameters...")
hyperparameters_dict = {}
for channel in spend_cols:
    logger.info(f"Configuring hyperparameters for channel: {channel}")
    if channel in ["Comms_spend", "Sponsorship_spend", "Offline_spend"]:
        logger.info(f"Using higher thetas [0.3, 0.9] for {channel}")
        thetas = [0.3, 0.9] if channel != "Comms_spend" else [0.2, 0.9]
    else:
        logger.info(f"Using default thetas [0, 0.3] for {channel}")
        thetas = [0, 0.3] if channel != "Digital_spend" else [0, 0.4]

    hyperparameters_dict[channel] = ChannelHyperparameters(
        alphas=[0.5, 3],
        gammas=[0.3, 1],
        thetas=thetas,
    )

hyperparameters = Hyperparameters(
    hyperparameters=hyperparameters_dict,
    adstock=AdstockType.GEOMETRIC,
    lambda_=[0, 1],
    train_size=[0.8, 1.0],
)
logger.info("Hyperparameters configuration completed")

# Initialize Robyn
logger.info(f"Initializing Robyn with working directory: {WORKING_DIR}")
robyn = Robyn(working_dir=WORKING_DIR, console_log_level="DEBUG")
logger.info("Robyn initialized successfully")

# Feature engineering
logger.info("Starting feature engineering process...")
from robyn.modeling.feature_engineering import FeatureEngineering

feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)
logger.info("Feature engineering object created")

featurized_mmm_data = feature_engineering.perform_feature_engineering()
logger.info("Feature engineering completed successfully")

# Model execution setup
logger.info("Setting up model execution...")
from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.model_executor import ModelExecutor

iterations = 10
trials = 1
TEST_VAL_ROWS = 10
logger.info(f"Configuring with {iterations} iterations and {trials} trials")

trials_config = TrialsConfig(iterations=iterations, trials=trials)
logger.info("Trials configuration created")

model_executor = ModelExecutor(
    mmmdata=mmm_data,
    holidays_data=holidays_data,
    hyperparameters=hyperparameters,
    calibration_input=None,
    featurized_mmm_data=featurized_mmm_data,
)
logger.info("Model executor created")

# Run the model
logger.info("Executing model run with Ridge regression...")
output_models = model_executor.model_run(
    trials_config=trials_config,
    ts_validation=True,
    add_penalty_factor=False,
    rssd_zero_penalty=True,
    nevergrad_algo=NevergradAlgorithm.TWO_POINTS_DE,
    objective_weights=[2, 1],
    model_name=Models.RIDGE,
    cores=16,
    seed=[42],
    val_size=TEST_VAL_ROWS // 2,
    test_size=TEST_VAL_ROWS // 2,
)
logger.info("Model execution completed successfully")

# Pareto optimization
logger.info("Starting Pareto optimization...")
from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer

pareto_optimizer = ParetoOptimizer(
    mmm_data, output_models, hyperparameters, featurized_mmm_data, holidays_data
)
logger.info("Pareto optimizer created")

pareto_result = pareto_optimizer.optimize(pareto_fronts="auto", min_candidates=1)
logger.info(
    f"Pareto optimization completed with {len(pareto_result.result_hyp_param)} candidates"
)

# Find best model by MAE
logger.info("Evaluating best model based on MAE...")
best_mae_row = (
    pareto_result.result_hyp_param[
        pareto_result.result_hyp_param["robynPareto"] <= pareto_result.pareto_fronts
    ]
    .sort_values("mae", ascending=True)
    .iloc[0]
)
best_mae_model = best_mae_row.sol_id
MAE_test_val = best_mae_row.mae
logger.info(f"Best model ID: {best_mae_model} with MAE: {MAE_test_val:.4f}")

# Get plot data and calculate metrics
logger.info("Extracting time series data for evaluation...")
plot_data = pareto_result.plot_data_collect[best_mae_model]
ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"]

# Transform to wide format for analysis
logger.info("Transforming time series data for analysis...")
wide_df = ts_data.pivot(index="ds", columns="variable", values="value").reset_index()
wide_df.columns.name = None
wide_df = wide_df.sort_values("ds")

# Analyze last rows (test + val )
last_rows = wide_df.tail(TEST_VAL_ROWS)
logger.info("\nAnalyzing last 10 rows of data:")
logger.info(f"\n{last_rows}")

# Calculate additional metrics for evaluation
metrics = {}
MAE_plot_data = mean_absolute_error(last_rows["actual"], last_rows["predicted"])
RMSE_plot_data = np.sqrt(
    mean_squared_error(last_rows["actual"], last_rows["predicted"])
)
r2_plot_data = r2_score(last_rows["actual"], last_rows["predicted"])

logger.info(f"Calculated MAE from plot data: {MAE_plot_data:.4f}")
logger.info(f"Calculated RMSE from plot data: {RMSE_plot_data:.4f}")
logger.info(f"Calculated R-squared from plot data: {r2_plot_data:.4f}")

# Validate MAE consistency with detailed logging
rtol = 0.05  # Relative tolerance (5%)
abs_diff = abs(MAE_plot_data - MAE_test_val)
rel_diff = abs_diff / MAE_test_val * 100 if MAE_test_val != 0 else float("inf")

logger.info(f"Validating MAE consistency...")
logger.info(f"Model MAE: {MAE_test_val:.4f}")
logger.info(f"Plot data MAE: {MAE_plot_data:.4f}")
logger.info(f"Absolute difference: {abs_diff:.4f}")
logger.info(f"Relative difference: {rel_diff:.2f}%")
logger.info(f"Relative tolerance: {rtol*100:.1f}%")

is_close = np.isclose(MAE_plot_data, MAE_test_val, rtol=rtol)
if is_close:
    logger.info(
        f"✓ MAE validation PASSED: Values are within {rtol*100:.1f}% relative tolerance"
    )
else:
    logger.warning(
        f"✗ MAE validation FAILED: Values differ by {rel_diff:.2f}%, exceeding tolerance of {rtol*100:.1f}%"
    )

# Enhanced assertion with detailed error message
assert is_close, (
    f"MAE validation failed: calculated MAE ({MAE_plot_data:.4f}) vs model MAE ({MAE_test_val:.4f}), "
    f"absolute difference: {abs_diff:.4f}, relative difference: {rel_diff:.2f}%, "
    f"exceeding tolerance of {rtol*100:.1f}%"
)

logger.info("MMM analysis completed successfully!")
