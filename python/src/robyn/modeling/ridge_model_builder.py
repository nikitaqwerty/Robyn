# pyre-strict

import json
import logging
import time
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import nevergrad as ng
import numpy as np
import pandas as pd
from nevergrad.optimization.base import Optimizer
from robyn.data.entities.calibration_input import CalibrationInput
from robyn.data.entities.holidays_data import HolidaysData
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.convergence.convergence import Convergence
from robyn.modeling.entities.enums import NevergradAlgorithm
from robyn.modeling.entities.model_refit_output import ModelRefitOutput
from robyn.modeling.entities.modeloutputs import ModelOutputs, Trial
from robyn.modeling.entities.modelrun_trials_config import TrialsConfig
from robyn.modeling.feature_engineering import FeaturizedMMMData
from robyn.modeling.ridge.models.ridge_utils import create_ridge_model_python
from robyn.modeling.ridge.ridge_data_builder import RidgeDataBuilder
from robyn.modeling.ridge.ridge_evaluate_model import RidgeModelEvaluator
from robyn.modeling.ridge.ridge_metrics_calculator import RidgeMetricsCalculator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


class RidgeModelBuilder:
    def __init__(
        self,
        mmm_data: MMMData,
        holiday_data: HolidaysData,
        calibration_input: CalibrationInput,
        hyperparameters: Hyperparameters,
        featurized_mmm_data: FeaturizedMMMData,
    ):
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.calibration_input = calibration_input
        self.hyperparameters = hyperparameters
        self.featurized_mmm_data = featurized_mmm_data

        # Initialize builders and calculators
        self.ridge_metrics_calculator = RidgeMetricsCalculator(
            mmm_data, hyperparameters
        )
        self.ridge_data_builder = RidgeDataBuilder(
            mmm_data, featurized_mmm_data, self.ridge_metrics_calculator
        )
        self.ridge_model_evaluator = RidgeModelEvaluator(
            self.mmm_data,
            self.featurized_mmm_data,
            self.ridge_metrics_calculator,
            self.ridge_data_builder,
            self.calibration_input,
            self.holiday_data,
        )
        self.logger = logging.getLogger(__name__)

    def initialize_nevergrad_optimizer(
        self,
        hyper_collect: Dict[str, Any],
        iterations: int,
        cores: int,
        nevergrad_algo: NevergradAlgorithm,
        calibration_input: Optional[Any] = None,
        objective_weights: Optional[List[float]] = None,
    ) -> Tuple[Optimizer, List[float]]:
        """Initialize Nevergrad optimizer exactly like R's implementation"""

        # Get number of hyperparameters
        hyper_count = len(hyper_collect["hyper_bound_list_updated"])
        self.logger.debug(f"Number of hyperparameters: {hyper_count}")

        # Create tuple for shape
        shape_tuple = (hyper_count,)
        self.logger.debug(f"Created shape tuple: {shape_tuple}")

        # Create instrumentation
        instrum = ng.p.Array(shape=shape_tuple, lower=0, upper=1)
        self.logger.debug(f"Created instrumentation: {instrum}")

        # Initialize optimizer
        optimizer = ng.optimizers.registry[nevergrad_algo.value](
            instrum, budget=iterations, num_workers=1
        )
        self.logger.debug(f"Initialized optimizer: {optimizer}")

        # Set multi-objective dimensions exactly like R
        try:
            if calibration_input is None:
                optimizer.tell(ng.p.MultiobjectiveReference(), (1, 1))
                if objective_weights is None:
                    objective_weights = [1, 1]
                optimizer.set_objective_weights(tuple(objective_weights[:2]))
            else:
                optimizer.tell(ng.p.MultiobjectiveReference(), (1, 1, 1))
                if objective_weights is None:
                    objective_weights = [1, 1, 1]
                optimizer.set_objective_weights(tuple(objective_weights[:3]))
        except Exception as e:
            self.logger.error(f"Error setting objective weights: {e}")

        # Log step5 nevergrad setup
        self.logger.debug(
            json.dumps(
                {
                    "step": "step5_nevergrad_setup",
                    "data": {
                        "iterations": {
                            "total": iterations,
                            "parallel": 1,  # Single process for now
                            "nevergrad": iterations,  # All iterations run sequentially
                        },
                        "optimizer": {
                            "name": nevergrad_algo.value,
                            "hyper_fixed": False,
                            "tuple_size": len(
                                hyper_collect["hyper_bound_list_updated"]
                            ),
                            "num_workers": 1,  # Single worker
                            "budget": iterations,
                        },
                        "objective": {
                            "has_calibration": calibration_input is not None,
                            "weights": objective_weights,
                            "dimensions": 3 if calibration_input else 2,
                        },
                        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[
                            :-3
                        ],
                    },
                },
                indent=2,
            )
        )

        return optimizer, objective_weights

    def build_models(
        self,
        trials_config: TrialsConfig,
        dt_hyper_fixed: Optional[pd.DataFrame] = None,
        ts_validation: bool = False,
        add_penalty_factor: bool = False,
        seed: List[int] = [123],
        rssd_zero_penalty: bool = True,
        objective_weights: Optional[List[float]] = None,
        nevergrad_algo: NevergradAlgorithm = NevergradAlgorithm.TWO_POINTS_DE,
        intercept: bool = True,
        intercept_sign: str = "non_negative",
        cores: Optional[int] = None,
        val_size: int = 5,
        test_size: int = 5,
        fixed_coefficients: Optional[Dict[str, float]] = None,
        fixed_intercept: Optional[float] = None,
        reinit_nevergrad_between_trials: bool = False,
        cv_n_folds: Optional[int] = None,
        cv_train_size: Optional[int] = None,
        hp_opt_score_target: str = "nrmse_train",
        observation_weights: Optional[np.ndarray] = None,
        coefficient_lower_limits: Optional[Dict[str, float]] = None,
        coefficient_upper_limits: Optional[Dict[str, float]] = None,
        intercept_lower_limit: Optional[float] = None,
        intercept_upper_limit: Optional[float] = None,
    ) -> ModelOutputs:
        start_time = time.time()

        # Initialize hyperparameters with flattened structure
        hyper_collect = self.ridge_data_builder._hyper_collector(
            self.hyperparameters,
            ts_validation,
            add_penalty_factor,
            dt_hyper_fixed,
            cores,
        )

        # Initialize Nevergrad optimizer once if not reinitializing between trials
        optimizer = None
        objective_weights_eff = None
        if not reinit_nevergrad_between_trials:
            optimizer, objective_weights_eff = self.initialize_nevergrad_optimizer(
                hyper_collect=hyper_collect,
                iterations=trials_config.iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                calibration_input=self.calibration_input,
                objective_weights=objective_weights,
            )

        # Run trials
        trials = []
        for trial in range(1, trials_config.trials + 1):
            # Reinitialize optimizer for each trial if requested
            if reinit_nevergrad_between_trials:
                optimizer, objective_weights_eff = self.initialize_nevergrad_optimizer(
                    hyper_collect=hyper_collect,
                    iterations=trials_config.iterations,
                    cores=cores,
                    nevergrad_algo=nevergrad_algo,
                    calibration_input=self.calibration_input,
                    objective_weights=objective_weights,
                )

            trial_result = self.ridge_model_evaluator._run_nevergrad_optimization(
                optimizer=optimizer,
                hyper_collect=hyper_collect,
                iterations=trials_config.iterations,
                cores=cores,
                nevergrad_algo=nevergrad_algo,
                intercept=intercept,
                intercept_sign=intercept_sign,
                ts_validation=ts_validation,
                add_penalty_factor=add_penalty_factor,
                objective_weights=objective_weights_eff,
                dt_hyper_fixed=dt_hyper_fixed,
                rssd_zero_penalty=rssd_zero_penalty,
                trial=trial,
                seed=seed[0] + trial,
                total_trials=trials_config.trials,
                val_size=val_size,
                test_size=test_size,
                fixed_coefficients=fixed_coefficients,
                fixed_intercept=fixed_intercept,
                cv_n_folds=cv_n_folds,
                cv_train_size=cv_train_size,
                hp_opt_score_target=hp_opt_score_target,
                observation_weights=observation_weights,
                coefficient_lower_limits=coefficient_lower_limits,
                coefficient_upper_limits=coefficient_upper_limits,
                intercept_lower_limit=intercept_lower_limit,
                intercept_upper_limit=intercept_upper_limit,
            )
            trials.append(trial_result)
        # Calculate convergence
        convergence = Convergence()
        convergence_results = convergence.calculate_convergence(trials)

        # Aggregate results with explicit type casting
        all_result_hyp_param = pd.concat(
            [trial.result_hyp_param for trial in trials], ignore_index=True
        )
        # Ensure mae_val and mape_val columns exist
        for col in ["mae_val", "mape_val"]:
            if col not in all_result_hyp_param.columns:
                all_result_hyp_param[col] = np.nan
        all_result_hyp_param = self.ridge_data_builder.safe_astype(
            all_result_hyp_param,
            {
                "sol_id": "str",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
                "nrmse": "float64",
                "decomp.rssd": "float64",
                "mape": "int64",
                "pos": "int64",
                "lambda": "float64",
                "lambda_hp": "float64",
                "lambda_max": "float64",
                "lambda_min_ratio": "float64",
                "rsq_train": "float64",
                "rsq_val": "float64",
                "rsq_test": "float64",
                "nrmse_train": "float64",
                "nrmse_val": "float64",
                "nrmse_test": "float64",
                "ElapsedAccum": "float64",
                "Elapsed": "float64",
                "mae_val": "float64",
                "mape_val": "float64",
                "train_size": "float64",
            },
        )

        all_x_decomp_agg = pd.concat(
            [trial.x_decomp_agg for trial in trials], ignore_index=True
        )
        # Ensure mae_val and mape_val columns exist
        for col in ["mae_val", "mape_val"]:
            if col not in all_x_decomp_agg.columns:
                all_x_decomp_agg[col] = np.nan
        all_x_decomp_agg = self.ridge_data_builder.safe_astype(
            all_x_decomp_agg,
            {
                "rn": "str",
                "coef": "float64",
                "xDecompAgg": "float64",
                "xDecompPerc": "float64",
                "xDecompMeanNon0": "float64",
                "xDecompMeanNon0Perc": "float64",
                "xDecompAggRF": "float64",
                "xDecompPercRF": "float64",
                "xDecompMeanNon0RF": "float64",
                "xDecompMeanNon0PercRF": "float64",
                "sol_id": "str",
                "pos": "bool",
                "mape": "int64",
                "mae_val": "float64",
                "mape_val": "float64",
            },
        )

        all_decomp_spend_dist = pd.concat(
            [
                trial.decomp_spend_dist
                for trial in trials
                if trial.decomp_spend_dist is not None
            ],
            ignore_index=True,
        )
        all_decomp_spend_dist = self.ridge_data_builder.safe_astype(
            all_decomp_spend_dist,
            {
                "rn": "str",
                "coef": "float64",
                "total_spend": "float64",
                "mean_spend": "float64",
                "effect_share": "float64",
                "spend_share": "float64",
                "xDecompAgg": "float64",
                "xDecompPerc": "float64",
                "xDecompMeanNon0": "float64",
                "xDecompMeanNon0Perc": "float64",
                "sol_id": "str",
                "pos": "bool",
                "mape": "int64",
                "nrmse": "float64",
                "decomp.rssd": "float64",
                "trial": "int64",
                "iterNG": "int64",
                "iterPar": "int64",
            },
        )
        # Convert hyper_bound_ng from dict to DataFrame
        hyper_bound_ng_df = pd.DataFrame()
        for param_name, bounds in hyper_collect["hyper_bound_list_updated"].items():
            hyper_bound_ng_df.loc[0, param_name] = bounds[0]
            hyper_bound_ng_df[param_name] = hyper_bound_ng_df[param_name].astype(
                "float64"
            )
        if "lambda" in hyper_bound_ng_df.columns:
            hyper_bound_ng_df["lambda"] = hyper_bound_ng_df["lambda"].astype("int64")
        # Create ModelOutputs
        model_outputs = ModelOutputs(
            trials=trials,
            train_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            cores=cores,
            iterations=trials_config.iterations,
            intercept=intercept,
            intercept_sign=intercept_sign,
            nevergrad_algo=nevergrad_algo,
            ts_validation=ts_validation,
            add_penalty_factor=add_penalty_factor,
            hyper_updated=hyper_collect["hyper_list_all"],
            hyper_fixed=hyper_collect["all_fixed"],
            convergence=convergence_results,
            select_id=self._select_best_model(trials),
            seed=seed,
            hyper_bound_ng=hyper_bound_ng_df,
            hyper_bound_fixed=hyper_collect["hyper_bound_list_fixed"],
            ts_validation_plot=None,
            all_result_hyp_param=all_result_hyp_param,
            all_x_decomp_agg=all_x_decomp_agg,
            all_decomp_spend_dist=all_decomp_spend_dist,
        )

        return model_outputs

    def _select_best_model(self, output_models: List[Trial]) -> str:
        # Extract relevant metrics
        nrmse_values = np.array([trial.nrmse for trial in output_models])
        decomp_rssd_values = np.array([trial.decomp_rssd for trial in output_models])

        # Normalize the metrics
        nrmse_norm = (nrmse_values - np.min(nrmse_values)) / (
            np.max(nrmse_values) - np.min(nrmse_values)
        )
        decomp_rssd_norm = (decomp_rssd_values - np.min(decomp_rssd_values)) / (
            np.max(decomp_rssd_values) - np.min(decomp_rssd_values)
        )

        # Calculate the combined score (assuming equal weights)
        combined_score = nrmse_norm + decomp_rssd_norm

        # Find the index of the best model (lowest combined score)
        best_index = np.argmin(combined_score)

        # Return the sol_id of the best model (changed from solID)
        return output_models[best_index].result_hyp_param["sol_id"].values[0]
