import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

version = "0.5.1tst"
data_path = Path(f"/Users/nrotanov/pari/mmm/data/{'.'.join(version.split('.')[0:2])}")
data_path.mkdir(parents=True, exist_ok=True)
WORKING_DIR = f"/Users/nrotanov/pari/mmm/output/robyn_output_{version}"

START_DATE = "2022-05-01"
END_DATE = "2025-01-01"

from robyn.robyn import Robyn
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters, ChannelHyperparameters
from robyn.data.entities.enums import AdstockType, DependentVarType

df = pd.read_csv(data_path.joinpath("weekly_data.csv"))

df = df.rename(columns={"FTD": "dep_var"})
df.Date = pd.to_datetime(df.Date)

spend_cols = [col for col in df.columns if col.endswith("_spend")]
event_cols = [col for col in df.columns if col.endswith("_event")]

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

dt_prophet_holidays = pd.read_csv(f"{data_path.parent}/dt_prophet_holidays.csv")
holidays_data = HolidaysData(
    dt_holidays=dt_prophet_holidays,
    prophet_vars=["trend", "season", "holiday"],
    prophet_country="RU",
    prophet_signs=["default", "default", "default"],
)

hyperparameters = Hyperparameters(
    hyperparameters={
        "AffiliatesCPA_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0, 0.3],
        ),
        "Comms_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0.2, 0.9],
        ),
        "Digital_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0, 0.4],
        ),
        "Mobile_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0, 0.3],
        ),
        "Sponsorship_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0.3, 0.9],
        ),
        "Offline_spend": ChannelHyperparameters(
            alphas=[0.5, 3],
            gammas=[0.3, 1],
            thetas=[0.3, 0.9],
        ),
    },
    adstock=AdstockType.GEOMETRIC,
    lambda_=[0, 1],
    train_size=[0.8, 1.0],
)

robyn = Robyn(working_dir=WORKING_DIR, console_log_level="DEBUG")

from robyn.modeling.feature_engineering import FeatureEngineering

feature_engineering = FeatureEngineering(mmm_data, hyperparameters, holidays_data)

featurized_mmm_data = feature_engineering.perform_feature_engineering()

from robyn.modeling.entities.enums import Models, NevergradAlgorithm
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.model_executor import ModelExecutor

trials_config = TrialsConfig(iterations=10, trials=1)

model_executor = ModelExecutor(
    mmmdata=mmm_data,
    holidays_data=holidays_data,
    hyperparameters=hyperparameters,
    calibration_input=None,
    featurized_mmm_data=featurized_mmm_data,
)

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
    val_size=5,
    test_size=5,
)

from robyn.modeling.pareto.pareto_optimizer import ParetoOptimizer
import copy

# 3. Create ParetoOptimizer instance
pareto_optimizer = ParetoOptimizer(
    mmm_data, output_models, hyperparameters, featurized_mmm_data, holidays_data
)

# 4. Run optimize function
pareto_result = pareto_optimizer.optimize(pareto_fronts=1, min_candidates=1)

best_mae_row = pareto_result.result_hyp_param.sort_values("mae", ascending=True).iloc[0]
best_mae_model = best_mae_row.sol_id
print(best_mae_model, "mae", best_mae_row.mae)
plot_data = pareto_result.plot_data_collect[best_mae_model]
ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"]


wide_df = ts_data.pivot(index="ds", columns="variable", values="value").reset_index()
wide_df.columns.name = None
wide_df = wide_df.sort_values("ds")
last_10 = wide_df.tail(10)
metrics = {}
metrics["mae"] = mean_absolute_error(last_10["actual"], last_10["predicted"])
metrics["r2"] = r2_score(last_10["actual"], last_10["predicted"])

# Display the results for original data
print("Original Data - Metrics for the last 10 data points:")
print(f"MAE: {metrics['mae']:.4f}")
print(f"R2: {metrics['r2']:.4f}")

print("\nOriginal data (last 10 rows):")
print(last_10)
