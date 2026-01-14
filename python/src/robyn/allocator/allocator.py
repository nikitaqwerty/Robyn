import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.validation.mmmdata_utils import MMMDataUtils
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.feature_engineering import FeaturizedMMMData

from .constants import (
    ALGO_MMA_AUGLAG,
    ALGO_SLSQP_AUGLAG,
    SCENARIO_MAX_RESPONSE,
    SCENARIO_TARGET_EFFICIENCY,
)
from .entities.allocation_params import AllocatorParams
from .entities.allocation_result import AllocationResult, MainPoints, OptimOutData
from .entities.constraints import Constraints
from .entities.optimization_result import OptimizationResult
from .optimizer import eval_f, eval_g_eq, eval_g_ineq, run_optimization
from .response import calculate_response, which_usecase


class BudgetAllocator:
    """Budget allocation optimizer for MMM models."""

    def __init__(
        self,
        mmm_data: MMMData,
        featurized_mmm_data: FeaturizedMMMData,
        hyperparameters: Hyperparameters,
        pareto_result: ParetoResult,
        select_model: str,
        params: AllocatorParams,
    ):
        """
        Initialize BudgetAllocator with MMM data and parameters.

        Args:
            mmm_data: MMM data object containing model inputs
            featurized_mmm_data: Processed MMM data
            hyperparameters: Model hyperparameters
            pareto_result: Pareto optimization results
            select_model: Selected model ID
            params: Allocation parameters
        """
        self.logger = logging.getLogger(__name__)
        self.mmm_data = mmm_data
        self.featurized_mmm_data = featurized_mmm_data
        self.hyperparameters = hyperparameters
        self.pareto_result = pareto_result
        self.select_model = select_model
        self.params = params
        self.params.quiet = True

        # Initialize optimization components
        self._setup_local_data_and_params()
        self._setup_spend_values()
        self._calculate_response_values()
        self._set_initial_values_and_bounds()
        self._setup_optimization_bounds_and_channels()
        self._setup_optimization_parameters()

        # Run optimization and get results
        self.dt_optim_out = self._optimize()

        # Calculate curves and main points
        curves_data = self._calculate_curves(self.dt_optim_out)
        scurve_data = self._prepare_scurve_data(
            curves_data["dt_optim_out_scurve"], curves_data["levs1"]
        )

        # Create and store allocation result
        self.result = AllocationResult(
            dt_optimOut=self.dt_optim_out,
            mainPoints=scurve_data["main_points"],
            scenario=self.params.scenario,
            usecase=self.usecase,
            total_budget=(
                self.params.total_budget
                if self.params.total_budget is not None
                else self.total_budget_window
            ),
            skipped_coef0=self.zero_coef_channel,
            skipped_constr=self.zero_constraint_channel,
            no_spend=self.zero_spend_channel,
        )

    def optimize(self) -> AllocationResult:
        """
        Run the budget allocation optimization.

        Returns:
            AllocationResult object containing optimization results
        """
        # TODO: Implement optimization logic
        raise NotImplementedError("Optimization not implemented yet")

    def calculate_response_for_budget(
        self,
        budget_allocation: Union[float, Dict[str, float]],
        marginal_increment: float = 1.0,
    ) -> Dict[str, Union[float, pd.Series, Dict[str, float]]]:
        """
        Calculate response for a given budget allocation, separating marginal and carryover responses.

        Args:
            budget_allocation: Either a single float (total budget to distribute proportionally)
                              or a dict mapping channel names to budget amounts
            marginal_increment: Increment used to calculate marginal response (default: 1.0)

        Returns:
            Dictionary containing:
                - 'total_response': Total response including carryover (pd.Series per channel)
                - 'marginal_response': Marginal response from incremental spend (pd.Series per channel)
                - 'carryover_response': Response from historical carryover only (pd.Series per channel)
                - 'total_response_sum': Sum of total responses across all channels (float)
                - 'marginal_response_sum': Sum of marginal responses across all channels (float)
                - 'carryover_response_sum': Sum of carryover responses across all channels (float)
                - 'budget_allocation': The budget allocation used (dict)
        """
        # Handle budget allocation input
        if isinstance(budget_allocation, (int, float)):
            # Distribute total budget proportionally based on initial spend shares
            total_budget = float(budget_allocation)
            budget_dict = {
                channel: total_budget * self.init_spend_share[channel]
                for channel in self.channel_for_allocation
            }
        elif isinstance(budget_allocation, dict):
            budget_dict = budget_allocation.copy()
            # Ensure all channels are included, set to 0 if not specified
            for channel in self.channel_for_allocation:
                if channel not in budget_dict:
                    budget_dict[channel] = 0.0
        else:
            raise TypeError(
                "budget_allocation must be a float (total budget) or dict (channel->budget mapping)"
            )

        # Initialize result dictionaries
        total_response = pd.Series(index=self.media_spend_sorted, dtype=float)
        marginal_response = pd.Series(index=self.media_spend_sorted, dtype=float)
        carryover_response = pd.Series(index=self.media_spend_sorted, dtype=float)

        # Calculate responses for each channel
        for channel in self.media_spend_sorted:
            if channel not in self.channel_for_allocation:
                # Skip excluded channels (zero coefficient or zero constraint)
                total_response[channel] = 0.0
                marginal_response[channel] = 0.0
                carryover_response[channel] = 0.0
                continue

            # Get channel-specific parameters
            channel_budget = budget_dict.get(channel, 0.0)
            hist_carryover_mean = np.mean(self.hist_carryover[channel])

            # Use coefs_sorted for consistency (has all channels)
            coeff = self.coefs_sorted[channel]
            alpha = self.alphas_eval[f"{channel}_alphas"]
            inflexion = self.inflexions_eval[f"{channel}_gammas"]
            theta = self.thetas[channel]

            # Calculate total response (with carryover)
            resp_total = self._fx_objective(
                x=channel_budget,
                coeff=coeff,
                alpha=alpha,
                inflexion=inflexion,
                x_hist_carryover=hist_carryover_mean,
                theta=theta,
                get_sum=True,
            )

            # Calculate carryover response only (no new spend, only historical carryover)
            resp_carryover_only = self._fx_objective(
                x=0.0,  # No new spend
                coeff=coeff,
                alpha=alpha,
                inflexion=inflexion,
                x_hist_carryover=hist_carryover_mean,
                theta=theta,
                get_sum=True,
            )

            # Calculate marginal response (incremental response from additional spend)
            resp_marginal = (
                self._fx_objective(
                    x=channel_budget + marginal_increment,
                    coeff=coeff,
                    alpha=alpha,
                    inflexion=inflexion,
                    x_hist_carryover=hist_carryover_mean,
                    theta=theta,
                    get_sum=True,
                )
                - resp_total
            )

            # Store results
            total_response[channel] = resp_total
            marginal_response[channel] = resp_marginal / marginal_increment
            carryover_response[channel] = resp_carryover_only

        # Calculate sums
        total_response_sum = total_response.sum()
        marginal_response_sum = marginal_response.sum()
        carryover_response_sum = carryover_response.sum()

        return {
            "total_response": total_response,
            "marginal_response": marginal_response,
            "carryover_response": carryover_response,
            "total_response_sum": float(total_response_sum),
            "marginal_response_sum": float(marginal_response_sum),
            "carryover_response_sum": float(carryover_response_sum),
            "budget_allocation": budget_dict,
        }

    def _setup_local_data_and_params(self) -> None:
        """Set up local data and parameters for the allocator"""
        # Get paid media spends and sort them
        paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends
        media_order = np.argsort(paid_media_spends)  # equivalent to R's order()
        self.media_spend_sorted = [paid_media_spends[i] for i in media_order]

        # Get dependent variable type
        self.dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type

        # Set up channel constraints
        if self.params.channel_constr_low is None:
            if self.params.scenario == SCENARIO_MAX_RESPONSE:
                self.params.channel_constr_low = 0.5
            elif self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
                self.params.channel_constr_low = 0.1

        if self.params.channel_constr_up is None:
            if self.params.scenario == SCENARIO_MAX_RESPONSE:
                self.params.channel_constr_up = 2
            elif self.params.scenario == SCENARIO_TARGET_EFFICIENCY:
                self.params.channel_constr_up = 10

        # Replicate single values if needed
        if isinstance(self.params.channel_constr_low, (int, float)):
            self.params.channel_constr_low = [self.params.channel_constr_low] * len(
                paid_media_spends
            )
        if isinstance(self.params.channel_constr_up, (int, float)):
            self.params.channel_constr_up = [self.params.channel_constr_up] * len(
                paid_media_spends
            )

        # Create Series with media names as index
        self.channel_constr_low = pd.Series(
            self.params.channel_constr_low, index=paid_media_spends
        ).reindex(self.media_spend_sorted)

        self.channel_constr_up = pd.Series(
            self.params.channel_constr_up, index=paid_media_spends
        ).reindex(self.media_spend_sorted)

        # Try with sol_id first
        self.dt_hyppar = self.pareto_result.result_hyp_param[
            self.pareto_result.result_hyp_param["sol_id"] == self.select_model
        ]
        self.dt_best_coef = self.pareto_result.x_decomp_agg[
            (self.pareto_result.x_decomp_agg["sol_id"] == self.select_model)
            & (self.pareto_result.x_decomp_agg["rn"].isin(paid_media_spends))
        ]

        # Sort media coefficients
        dt_coef = self.dt_best_coef[["rn", "coef"]].copy()
        get_rn_order = dt_coef["rn"].argsort()
        self.dt_coef_sorted = dt_coef.iloc[get_rn_order].reset_index(drop=True)
        self.dt_best_coef = self.dt_best_coef.iloc[get_rn_order].reset_index(drop=True)

        # Create coefficient selector (coef > 0)
        self.coef_selector_sorted = pd.Series(
            self.dt_coef_sorted["coef"] > 0, index=self.dt_coef_sorted["rn"]
        )

        # Filter hyperparameters for media channels
        hyper_columns = self._get_hyper_names(
            self.hyperparameters.adstock, self.media_spend_sorted
        )
        self.dt_hyppar = self.dt_hyppar[sorted(hyper_columns)]

        # Filter best coefficients for media spend sorted
        self.dt_best_coef = self.dt_best_coef[
            self.dt_best_coef["rn"].isin(self.media_spend_sorted)
        ]

        # Sort channel constraints to match media order
        self.channel_constr_low_sorted = self.channel_constr_low[
            self.media_spend_sorted
        ]
        self.channel_constr_up_sorted = self.channel_constr_up[self.media_spend_sorted]

        # Get hill parameters
        hill_params = self._get_hill_params()
        self.alphas = hill_params["alphas"]
        self.inflexions = hill_params["inflexions"]
        self.coefs_sorted = hill_params["coefs_sorted"]

        # Log all hill parameters
        self.logger.debug(
            json.dumps(
                {
                    "step": "hill_parameters",
                    "data": {
                        "alphas": self.alphas.to_dict(),
                        "inflexions": self.inflexions.to_dict(),
                        "coefs_sorted": self.coefs_sorted.to_dict(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _setup_spend_values(self) -> None:
        """Set up spend values based on date range"""
        # Get window indices
        window_start = self.mmm_data.mmmdata_spec.rolling_window_start_which
        window_end = self.mmm_data.mmmdata_spec.rolling_window_end_which
        self.window_loc = slice(window_start, window_end + 1)

        # Get optimized cost data for window
        self.dt_optim_cost = self.featurized_mmm_data.dt_mod.iloc[
            window_start : window_end + 1
        ].copy()

        # Check and update date range
        date_range = MMMDataUtils.check_metric_dates(
            date_range=self.params.date_range,
            all_dates=pd.Series(self.dt_optim_cost["ds"]),  # Convert to Series
            day_interval=self.mmm_data.mmmdata_spec.interval_type,
        )
        self.date_min = date_range["date_range_updated"][0]
        self.date_max = date_range["date_range_updated"][-1]

        # Validate date range
        MMMDataUtils.check_daterange(
            self.date_min, self.date_max, self.dt_optim_cost["ds"]
        )

        # Adjust date range if needed
        if self.date_min is None:
            self.date_min = self.dt_optim_cost["ds"].min()
        if self.date_max is None:
            self.date_max = self.dt_optim_cost["ds"].max()
        if self.date_min < self.dt_optim_cost["ds"].min():
            self.date_min = self.dt_optim_cost["ds"].min()
        if self.date_max > self.dt_optim_cost["ds"].max():
            self.date_max = self.dt_optim_cost["ds"].max()

        # Filter historical data by date range
        self.hist_filtered = self.dt_optim_cost[
            (self.dt_optim_cost["ds"] >= self.date_min)
            & (self.dt_optim_cost["ds"] <= self.date_max)
        ]

        # Calculate historical spend metrics for all data
        spend_cols = self.media_spend_sorted
        self.hist_spend_all = self.dt_optim_cost[spend_cols].sum()
        self.hist_spend_all_total = self.hist_spend_all.sum()
        self.hist_spend_all_unit = self.dt_optim_cost[spend_cols].mean()
        self.hist_spend_all_unit_total = self.hist_spend_all_unit.sum()
        self.hist_spend_all_share = (
            self.hist_spend_all_unit / self.hist_spend_all_unit_total
        )

        # Calculate historical spend metrics for window
        self.hist_spend_window = self.hist_filtered[spend_cols].sum()
        self.hist_spend_window_total = self.hist_spend_window.sum()
        self.hist_spend_window_unit = self.init_spend_unit = self.hist_filtered[
            spend_cols
        ].mean()
        self.hist_spend_window_unit_total = self.hist_spend_window_unit.sum()
        self.hist_spend_window_share = (
            self.hist_spend_window_unit / self.hist_spend_window_unit_total
        )

        # Calculate simulation period
        self.simulation_period = self.initial_mean_period = self.hist_filtered[
            spend_cols
        ].count()
        self.n_dates = {
            media: self.hist_filtered["ds"].values for media in self.media_spend_sorted
        }

        # Print date window info if not quiet
        if not self.params.quiet:
            print(
                f"Date Window: {self.date_min}:{self.date_max} "
                f"({self.initial_mean_period.iloc[0]} {self.mmm_data.mmmdata_spec.interval_type}s)"
            )

        # Identify zero spend channels
        self.zero_spend_channel = self.hist_spend_window[
            self.hist_spend_window == 0
        ].index.tolist()

        # Calculate initial spend metrics
        self.init_spend_unit_total = self.init_spend_unit.sum()
        self.init_spend_share = self.init_spend_unit / self.init_spend_unit_total

        # Calculate total budget
        period_length = self.initial_mean_period.iloc[0]
        self.total_budget_unit = (
            self.init_spend_unit_total
            if self.params.total_budget is None
            else self.params.total_budget / period_length
        )
        self.total_budget_window = self.total_budget_unit * period_length

        # Get use case based on inputs
        self.usecase = which_usecase(
            self.init_spend_unit.iloc[0], self.params.date_range
        )

        # Set ndates_loc based on use case
        if self.usecase == "all_historical_vec":
            self.ndates_loc = self.featurized_mmm_data.dt_mod[
                self.featurized_mmm_data.dt_mod["ds"].isin(self.hist_filtered["ds"])
            ].index.tolist()
        else:
            self.ndates_loc = list(range(len(self.hist_filtered["ds"])))

        # Add budget type to usecase
        budget_type = (
            "+ defined_budget"
            if self.params.total_budget is not None
            else "+ historical_budget"
        )
        self.usecase = f"{self.usecase} {budget_type}"

    def _get_hyper_names(self, adstock: str, media_list: List[str]) -> List[str]:
        """Get hyperparameter column names based on adstock type and media list"""
        base_params = ["alphas", "gammas"]
        if adstock == "geometric":
            base_params.append("thetas")
        elif "weibull" in adstock:
            base_params.extend(["shapes", "scales"])

        hyper_names = []
        for media in media_list:
            for param in base_params:
                hyper_names.append(f"{media}_{param}")

        return hyper_names

    def _get_hill_params(self) -> Dict[str, pd.Series]:
        """Get hill parameters for each channel"""
        # Get alpha and gamma parameters
        hill_params_cols = [
            col
            for col in self.dt_hyppar.columns
            if col.endswith("_alphas") or col.endswith("_gammas")
        ]
        hill_hyp_par = self.dt_hyppar[hill_params_cols].iloc[0]  # Get first row

        # Extract alphas and gammas for sorted media WITH suffixes
        alphas = pd.Series(
            [hill_hyp_par[f"{media}_alphas"] for media in self.media_spend_sorted],
            index=[
                f"{media}_alphas" for media in self.media_spend_sorted
            ],  # Keep suffixes
        )
        gammas = pd.Series(
            [hill_hyp_par[f"{media}_gammas"] for media in self.media_spend_sorted],
            index=[
                f"{media}_gammas" for media in self.media_spend_sorted
            ],  # Keep suffixes
        )

        # Get adstocked media data
        chn_adstocked = (
            self.pareto_result.media_vec_collect[
                (self.pareto_result.media_vec_collect["type"] == "adstockedMedia")
                & (self.pareto_result.media_vec_collect["sol_id"] == self.select_model)
            ][self.media_spend_sorted]
        ).iloc[
            self.mmm_data.mmmdata_spec.rolling_window_start_which : self.mmm_data.mmmdata_spec.rolling_window_end_which
            + 1
        ]

        # Calculate inflexions (with suffixes to match gammas)
        inflexions = pd.Series(
            index=[f"{media}_gammas" for media in self.media_spend_sorted], dtype=float
        )
        for i, media in enumerate(self.media_spend_sorted):
            # Get max value from adstocked media
            max_value = chn_adstocked[media].max()
            gamma = gammas[f"{media}_gammas"]
            inflexions[f"{media}_gammas"] = max_value * gamma

        # Get sorted coefficients (no suffix needed here)
        coefs_sorted = pd.Series(
            self.dt_coef_sorted["coef"].values, index=self.dt_coef_sorted["rn"]
        ).reindex(self.media_spend_sorted)

        return {
            "alphas": alphas,
            "inflexions": inflexions,
            "coefs_sorted": coefs_sorted,
        }

    # Modify the _calculate_response_values method to use the updated _fx_objective function
    def _calculate_response_values(self) -> None:
        """Calculate response values based on date range."""
        # Initialize response values
        self.init_response_unit = pd.Series(index=self.media_spend_sorted)
        self.init_response_marg_unit = pd.Series(index=self.media_spend_sorted)
        self.hist_carryover = {}
        self.qa_carryover = {}
        self.thetas = {}  # Add storage for theta values

        # Retrieve theta values from hyperparameters
        theta_params_cols = [
            col for col in self.dt_hyppar.columns if col.endswith("_thetas")
        ]
        theta_hyp_par = self.dt_hyppar[theta_params_cols].iloc[0]

        for media in self.media_spend_sorted:
            # Store theta parameter
            self.thetas[media] = theta_hyp_par[f"{media}_thetas"]

            response = calculate_response(
                mmm_data=self.mmm_data,
                featurized_mmm_data=self.featurized_mmm_data,
                pareto_result=self.pareto_result,
                hyperparameters=self.hyperparameters,
                select_model=self.select_model,
                metric_name=media,
                dt_hyppar=self.pareto_result.result_hyp_param,
                dt_coef=self.pareto_result.x_decomp_agg,
            )

            # Store carryover values with dates as index
            hist_carryover_temp = response["input_carryover"][self.window_loc]
            if isinstance(hist_carryover_temp, np.ndarray):
                hist_carryover_temp = pd.Series(
                    hist_carryover_temp, index=response["date"][self.window_loc]
                )
            self.hist_carryover[media] = hist_carryover_temp

            # Store QA carryover values (rounded total input for the window)
            self.qa_carryover[media] = np.round(
                response["input_total"][self.window_loc]
            )

            # Use initial spend unit for x_input - now passing theta parameter
            x_input = self.init_spend_unit[media]

            resp_simulate = self._fx_objective(
                x=x_input,
                coeff=self.coefs_sorted[media],
                alpha=self.alphas[f"{media}_alphas"],
                inflexion=self.inflexions[f"{media}_gammas"],
                x_hist_carryover=hist_carryover_temp.mean(),
                theta=self.thetas[media],  # Add theta parameter
                get_sum=True,
            )

            resp_simulate_plus1 = self._fx_objective(
                x=x_input + 1,
                coeff=self.coefs_sorted[media],
                alpha=self.alphas[f"{media}_alphas"],
                inflexion=self.inflexions[f"{media}_gammas"],
                x_hist_carryover=hist_carryover_temp.mean(),
                theta=self.thetas[media],  # Add theta parameter
                get_sum=True,
            )

            # Store response values
            self.init_response_unit[media] = resp_simulate
            self.init_response_marg_unit[media] = resp_simulate_plus1 - resp_simulate

    def _fx_objective(
        self,
        x: Union[float, np.ndarray],
        coeff: float,
        alpha: float,
        inflexion: float,
        x_hist_carryover: Union[float, np.ndarray],
        theta: float,  # Add theta parameter for proper adstocking
        get_sum: bool = True,
    ) -> float:
        """Calculate objective function value using proper adstock and Hill transformation."""
        # Convert to numpy arrays
        x_array = np.array([x]) if isinstance(x, (int, float)) else np.array(x)

        # Apply geometric adstock (same as training phase)
        x_decayed = np.zeros_like(x_array)
        x_decayed[0] = x_array[0]

        # Apply recursive geometric adstock
        for i in range(1, len(x_array)):
            x_decayed[i] = x_array[i] + theta * x_decayed[i - 1]

        # Add historical carryover effect as initial condition
        x_adstocked = x_decayed + x_hist_carryover

        # Hill transformation matching training code
        numerator = x_adstocked**alpha
        denominator = x_adstocked**alpha + inflexion**alpha
        x_saturated = numerator / denominator

        if get_sum:
            x_out = coeff * np.sum(x_saturated)
        else:
            x_out = coeff * x_saturated

        return x_out

    def _set_initial_values_and_bounds(self):
        """Set initial values and bounds for optimization."""

        # Calculate extended constraints
        channel_constr_low_sorted_ext = np.where(
            1
            - (1 - self.channel_constr_low_sorted)
            * self.params.channel_constr_multiplier
            < 0,
            0,
            1
            - (1 - self.channel_constr_low_sorted)
            * self.params.channel_constr_multiplier,
        )

        channel_constr_up_sorted_ext = np.where(
            1
            + (self.channel_constr_up_sorted - 1)
            * self.params.channel_constr_multiplier
            < 0,
            self.channel_constr_up_sorted * self.params.channel_constr_multiplier,
            1
            + (self.channel_constr_up_sorted - 1)
            * self.params.channel_constr_multiplier,
        )

        # Store extended constraints
        self.channel_constr_low_sorted_ext = channel_constr_low_sorted_ext
        self.channel_constr_up_sorted_ext = channel_constr_up_sorted_ext

        # Set target value extension
        self.target_value_ext = self.params.target_value
        if self.params.scenario == "target_efficiency":
            # Reset extended constraints for target efficiency
            self.channel_constr_low_sorted_ext = self.channel_constr_low_sorted
            self.channel_constr_up_sorted_ext = self.channel_constr_up_sorted

            if self.mmm_data.mmmdata_spec.dep_var_type == "conversion":
                if self.params.target_value is None:
                    self.params.target_value = (
                        np.sum(self.init_spend_unit) / np.sum(self.init_response_unit)
                    ) * 1.2
                self.target_value_ext = self.params.target_value * 1.5
            else:
                if self.params.target_value is None:
                    self.params.target_value = (
                        np.sum(self.init_response_unit) / np.sum(self.init_spend_unit)
                    ) * 0.8
                self.target_value_ext = 1

        # Set initial values
        self.temp_init = self.init_spend_unit.copy()
        self.temp_init_all = self.init_spend_unit.copy()

        # If no spend within window as initial spend, use historical average
        if len(self.zero_spend_channel) > 0:
            for channel in self.zero_spend_channel:
                self.temp_init_all[channel] = self.hist_spend_all_unit[channel]

    def _setup_optimization_bounds_and_channels(self):
        """Setup optimization bounds and handle channel exclusions."""

        # Convert to numpy arrays if needed
        self.media_spend_sorted = np.array(self.media_spend_sorted)
        self.channel_constr_low_sorted = np.array(self.channel_constr_low_sorted)
        self.channel_constr_up_sorted = np.array(self.channel_constr_up_sorted)

        # Set initial bounds
        self.temp_ub = self.temp_ub_all = self.channel_constr_up_sorted.copy()
        self.temp_lb = self.temp_lb_all = self.channel_constr_low_sorted.copy()
        self.temp_ub_ext = self.temp_ub_ext_all = (
            self.channel_constr_up_sorted_ext.copy()
        )
        self.temp_lb_ext = self.temp_lb_ext_all = (
            self.channel_constr_low_sorted_ext.copy()
        )

        # Calculate bounds (but not x0)
        self.lb_all = self.temp_init_all * self.temp_lb_all
        self.ub_all = self.temp_init_all * self.temp_ub_all
        self.lb_ext_all = self.temp_init_all * self.temp_lb_ext_all
        self.ub_ext_all = self.temp_init_all * self.temp_ub_ext_all

        # Set initial values for x0 - starting near lower bounds to force optimization
        self.x0_all = self.lb_all * 1.05  # Just 5% above lower bounds
        self.x0_ext_all = self.lb_ext_all * 1.05

        # Log initial bounds
        self.logger.debug(
            json.dumps(
                {
                    "step": "initial_bounds",
                    "data": {
                        "temp_ub": self.temp_ub.tolist(),
                        "temp_lb": self.temp_lb.tolist(),
                        "temp_ub_ext": self.temp_ub_ext.tolist(),
                        "temp_lb_ext": self.temp_lb_ext.tolist(),
                        "x0_all": self.x0_all.tolist(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

        # Identify channels to exclude
        skip_these = (self.channel_constr_low_sorted == 0) & (
            self.channel_constr_up_sorted == 0
        )

        self.zero_constraint_channel = self.media_spend_sorted[skip_these]

        # Handle zero coefficient channels
        if not all(self.coef_selector_sorted) and not self.params.keep_zero_coefs:
            self.zero_coef_channel = [
                channel
                for channel in self.media_spend_sorted
                if channel not in self.media_spend_sorted[self.coef_selector_sorted]
            ]
            if not self.params.quiet:
                print(
                    f"Excluded variables (coefficients are 0): {', '.join(self.zero_coef_channel)}"
                )
        else:
            self.zero_coef_channel = []

        # Determine channels to drop and keep for allocation
        channels_to_drop = np.isin(
            self.media_spend_sorted,
            np.concatenate([self.zero_coef_channel, self.zero_constraint_channel]),
        )
        self.channel_for_allocation = self.media_spend_sorted[~channels_to_drop]

        # Update bounds if any channels are dropped
        if any(channels_to_drop):
            self.temp_init = self.temp_init_all[self.channel_for_allocation]
            self.temp_ub = self.temp_ub_all[self.channel_for_allocation]
            self.temp_lb = self.temp_lb_all[self.channel_for_allocation]
            self.temp_ub_ext = self.temp_ub_ext_all[self.channel_for_allocation]
            self.temp_lb_ext = self.temp_lb_ext_all[self.channel_for_allocation]
            self.x0 = self.x0_all[self.channel_for_allocation]
            self.lb = self.lb_all[self.channel_for_allocation]
            self.ub = self.ub_all[self.channel_for_allocation]
            self.x0_ext = self.x0_ext_all[self.channel_for_allocation]
            self.lb_ext = self.lb_ext_all[self.channel_for_allocation]
            self.ub_ext = self.ub_ext_all[self.channel_for_allocation]

        # Final bound calculations
        self.lb = self.temp_init * self.temp_lb
        self.ub = self.temp_init * self.temp_ub
        self.lb_ext = self.temp_init * self.temp_lb_ext
        self.ub_ext = self.temp_init * self.temp_ub_ext

        # Generate a completely different starting point
        # Use a single random channel for majority of budget
        if len(self.channel_for_allocation) > 1:
            # Get total budget
            total_budget = self.total_budget_unit

            # Start with minimal allocations
            self.x0 = self.lb.copy()
            self.x0_ext = self.lb_ext.copy()

            # Calculate available budget after minimal allocations
            remaining_budget = total_budget - np.sum(self.x0)

            if remaining_budget > 0:
                # Find the highest ROI channel (if available) or use a random one
                high_roi_idx = 0
                try:
                    roi_values = {}
                    for i, channel in enumerate(self.channel_for_allocation):
                        spend = self.temp_init.iloc[i]
                        response = (
                            self.init_response_unit[channel]
                            if channel in self.init_response_unit.index
                            else 0
                        )
                        roi = response / spend if spend > 0 else 0
                        roi_values[i] = roi

                    high_roi_idx = max(roi_values, key=roi_values.get)
                except:
                    # If calculating ROI fails, just use the first channel
                    high_roi_idx = 0

                # Allocate remaining budget to the target channel (up to its upper bound)
                available_headroom = (
                    self.ub.iloc[high_roi_idx] - self.x0.iloc[high_roi_idx]
                )
                allocation = min(remaining_budget, available_headroom)
                self.x0.iloc[high_roi_idx] += allocation

                # If we still have remaining budget after maxing out the target channel,
                # distribute it proportionally among the other channels
                if (
                    remaining_budget - allocation > 0.01
                ):  # Small tolerance for floating point
                    secondary_budget = remaining_budget - allocation
                    other_channels = np.ones(len(self.x0), dtype=bool)
                    other_channels[high_roi_idx] = False

                    # Calculate available headroom for other channels
                    other_headroom = self.ub[other_channels] - self.x0[other_channels]
                    total_headroom = np.sum(other_headroom)

                    if total_headroom > 0:
                        # Distribute proportionally to available headroom
                        distribution_ratio = secondary_budget / total_headroom
                        additional_allocations = other_headroom * distribution_ratio
                        self.x0[other_channels] += additional_allocations

            # Do the same for extended bounds
            remaining_budget_ext = total_budget - np.sum(self.x0_ext)
            if remaining_budget_ext > 0:
                available_headroom_ext = (
                    self.ub_ext.iloc[high_roi_idx] - self.x0_ext.iloc[high_roi_idx]
                )
                allocation_ext = min(remaining_budget_ext, available_headroom_ext)
                self.x0_ext.iloc[high_roi_idx] += allocation_ext

                if remaining_budget_ext - allocation_ext > 0.01:
                    secondary_budget_ext = remaining_budget_ext - allocation_ext
                    other_channels = np.ones(len(self.x0_ext), dtype=bool)
                    other_channels[high_roi_idx] = False

                    other_headroom_ext = (
                        self.ub_ext[other_channels] - self.x0_ext[other_channels]
                    )
                    total_headroom_ext = np.sum(other_headroom_ext)

                    if total_headroom_ext > 0:
                        distribution_ratio_ext = (
                            secondary_budget_ext / total_headroom_ext
                        )
                        additional_allocations_ext = (
                            other_headroom_ext * distribution_ratio_ext
                        )
                        self.x0_ext[other_channels] += additional_allocations_ext

        # Ensure the initial point satisfies the budget constraint exactly
        total_x0 = np.sum(self.x0)
        if abs(total_x0 - self.total_budget_unit) > 1e-6:
            # Apply a scaling factor to match the budget exactly
            scaling_factor = self.total_budget_unit / total_x0
            self.x0 = self.x0 * scaling_factor

        total_x0_ext = np.sum(self.x0_ext)
        if abs(total_x0_ext - self.total_budget_unit) > 1e-6:
            scaling_factor_ext = self.total_budget_unit / total_x0_ext
            self.x0_ext = self.x0_ext * scaling_factor_ext

        # Log final bounds after dropping channels
        self.logger.debug(
            json.dumps(
                {
                    "step": "final_bounds",
                    "data": {
                        "channels": list(self.channel_for_allocation),
                        "temp_init": self.temp_init.tolist(),
                        "x0": self.x0.tolist(),
                        "x0_sum": float(np.sum(self.x0)),
                        "total_budget_unit": float(self.total_budget_unit),
                        "lb": self.lb.tolist(),
                        "ub": self.ub.tolist(),
                        "x0_ext": self.x0_ext.tolist(),
                        "lb_ext": self.lb_ext.tolist(),
                        "ub_ext": self.ub_ext.tolist(),
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _setup_optimization_parameters(self):
        """Setup optimization parameters and options for nloptr."""

        # Gather values for optimization - keep the suffixes as in R
        self.coefs_eval = {
            channel: self.coefs_sorted[channel]
            for channel in self.channel_for_allocation
        }

        # Keep the full parameter names with suffixes
        self.alphas_eval = self.alphas  # Already has _alphas suffix
        self.inflexions_eval = self.inflexions  # Already has _gammas suffix

        self.hist_carryover_eval = {
            channel: self.hist_carryover[channel]
            for channel in self.channel_for_allocation
        }
        # Create evaluation dictionary
        self.eval_dict = {
            "coefs_eval": self.coefs_eval,
            "alphas_eval": self.alphas_eval,
            "inflexions_eval": self.inflexions_eval,
            "total_budget": self.params.total_budget,
            "total_budget_unit": self.total_budget_unit,
            "hist_carryover_eval": self.hist_carryover_eval,
            "target_value": self.params.target_value,
            "target_value_ext": self.target_value_ext,
            "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
        }

        # Set optimization options based on algorithm
        if self.params.optim_algo == ALGO_MMA_AUGLAG:
            self.local_opts = {"algorithm": "NLOPT_LD_MMA", "xtol_rel": 1.0e-10}
        elif self.params.optim_algo == ALGO_SLSQP_AUGLAG:
            self.local_opts = {"algorithm": "NLOPT_LD_SLSQP", "xtol_rel": 1.0e-10}
        else:
            raise ValueError(
                f"Unsupported optimization algorithm: {self.params.optim_algo}"
            )
        # Log the dictionary with converted values
        self.logger.debug(
            json.dumps(
                {
                    "step": "eval_dict",
                    "data": {
                        "coefs_eval": (
                            list(self.coefs_eval.values())
                            if isinstance(self.coefs_eval, dict)
                            else self.coefs_eval.tolist()
                        ),
                        "alphas_eval": (
                            list(self.alphas_eval.values())
                            if isinstance(self.alphas_eval, dict)
                            else self.alphas_eval.tolist()
                        ),
                        "inflexions_eval": (
                            list(self.inflexions_eval.values())
                            if isinstance(self.inflexions_eval, dict)
                            else self.inflexions_eval.tolist()
                        ),
                        "total_budget": self.params.total_budget,
                        "total_budget_unit": float(self.total_budget_unit),
                        "hist_carryover_eval": {
                            k: v.tolist() if hasattr(v, "tolist") else float(v)
                            for k, v in self.hist_carryover_eval.items()
                        },
                        "target_value": self.params.target_value,
                        "target_value_ext": self.target_value_ext,
                        "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
                    },
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                indent=2,
            )
        )

    def _run_allocation_cycle(
        self, lower_bounds, upper_bounds, total_budget, num_cycles, x_hist_carryover
    ):
        """
        Run the iterative marginal response allocation cycle.

        Returns:
            tuple: (optimized_spend, last_allocation)
        """
        optimized_spend = {ch: lower_bounds[ch] for ch in self.channel_for_allocation}
        remaining_budget = total_budget - sum(optimized_spend.values())
        last_allocation = 0

        for cycle in range(num_cycles):
            self.logger.debug(f"Cycle {cycle + 1} of {num_cycles}")
            cycles_remaining = num_cycles - cycle
            allocation_this_cycle = remaining_budget / cycles_remaining
            last_allocation = allocation_this_cycle

            # Calculate marginal responses
            marginal_responses = {}
            for channel in self.channel_for_allocation:
                current_spend = optimized_spend[channel]
                current_response = self._fx_objective(
                    x=current_spend,
                    coeff=self.coefs_eval[channel],
                    alpha=self.alphas_eval[f"{channel}_alphas"],
                    inflexion=self.inflexions_eval[f"{channel}_gammas"],
                    x_hist_carryover=x_hist_carryover[channel],
                    theta=self.thetas[channel],
                    get_sum=True,
                )
                response_plus = self._fx_objective(
                    x=current_spend + allocation_this_cycle,
                    coeff=self.coefs_eval[channel],
                    alpha=self.alphas_eval[f"{channel}_alphas"],
                    inflexion=self.inflexions_eval[f"{channel}_gammas"],
                    x_hist_carryover=x_hist_carryover[channel],
                    theta=self.thetas[channel],
                    get_sum=True,
                )
                marginal_response = (
                    response_plus - current_response
                ) / allocation_this_cycle
                marginal_responses[channel] = marginal_response

            # Sort by marginal response
            sorted_channels = sorted(
                self.channel_for_allocation,
                key=lambda ch: marginal_responses[ch],
                reverse=True,
            )

            # Allocate budget
            allocation_remaining = allocation_this_cycle
            for channel in sorted_channels:
                if allocation_remaining <= 1e-6:
                    break
                headroom = upper_bounds[channel] - optimized_spend[channel]
                allocation = min(allocation_remaining, headroom)
                if allocation > 0:
                    optimized_spend[channel] += allocation
                    remaining_budget -= allocation
                    allocation_remaining -= allocation

            if remaining_budget > 1e-6:
                self.logger.warning(f"Unallocated budget remaining: {remaining_budget}")

        return optimized_spend, last_allocation

    def _optimize(self):
        """Run budget allocation with iterative marginal response estimation."""
        self.logger.debug(
            "Starting iterative allocation with marginal response estimation"
        )

        # Initial setup
        original_spend = pd.Series(
            {
                channel: self.temp_init.iloc[i]
                for i, channel in enumerate(self.channel_for_allocation)
            }
        )
        lower_bounds = {
            channel: self.lb.iloc[i]
            for i, channel in enumerate(self.channel_for_allocation)
        }
        upper_bounds = {
            channel: self.ub.iloc[i]
            for i, channel in enumerate(self.channel_for_allocation)
        }
        total_budget = self.total_budget_unit
        x_hist_carryover = {k: np.mean(v) for k, v in self.hist_carryover_eval.items()}
        num_cycles = 10  # Can be made configurable via params

        # --- Bounded Allocation ---
        optimized_spend, last_allocation = self._run_allocation_cycle(
            lower_bounds, upper_bounds, total_budget, num_cycles, x_hist_carryover
        )

        # Evaluate bounded allocation
        optimized_responses = {
            ch: self._fx_objective(
                x=optimized_spend[ch],
                coeff=self.coefs_eval[ch],
                alpha=self.alphas_eval[f"{ch}_alphas"],
                inflexion=self.inflexions_eval[f"{ch}_gammas"],
                x_hist_carryover=x_hist_carryover[ch],
                theta=self.thetas[ch],
                get_sum=True,
            )
            for ch in self.channel_for_allocation
        }

        # --- Unbounded Allocation ---
        extended_upper_bounds = {
            channel: upper * self.params.channel_constr_multiplier
            for channel, upper in upper_bounds.items()
        }

        optimized_spend_unbounded, last_allocation_unb = self._run_allocation_cycle(
            lower_bounds,
            extended_upper_bounds,
            total_budget,
            num_cycles,
            x_hist_carryover,
        )

        # Evaluate unbounded allocation
        optimized_responses_unbounded = {
            ch: self._fx_objective(
                x=optimized_spend_unbounded[ch],
                coeff=self.coefs_eval[ch],
                alpha=self.alphas_eval[f"{ch}_alphas"],
                inflexion=self.inflexions_eval[f"{ch}_gammas"],
                x_hist_carryover=x_hist_carryover[ch],
                theta=self.thetas[ch],
                get_sum=True,
            )
            for ch in self.channel_for_allocation
        }

        # --- Prepare Output ---
        optm_spend_unit = self.init_spend_unit.copy()
        optm_response_unit = self.init_response_unit.copy()
        optm_response_marg_unit = self.init_response_marg_unit.copy()
        optm_spend_unit_unbound = self.init_spend_unit.copy()
        optm_response_unit_unbound = self.init_response_unit.copy()
        optm_response_marg_unit_unbound = self.init_response_marg_unit.copy()

        for channel in self.channel_for_allocation:
            # Bounded allocation results
            optm_spend_unit[channel] = optimized_spend[channel]
            optm_response_unit[channel] = optimized_responses[channel]
            optm_response_marg_unit[channel] = (
                self._fx_objective(
                    x=optimized_spend[channel] + last_allocation,
                    coeff=self.coefs_eval[channel],
                    alpha=self.alphas_eval[f"{channel}_alphas"],
                    inflexion=self.inflexions_eval[f"{channel}_gammas"],
                    x_hist_carryover=x_hist_carryover[channel],
                    theta=self.thetas[channel],
                    get_sum=True,
                )
                - optimized_responses[channel]
            ) / last_allocation

            # Unbounded allocation results
            optm_spend_unit_unbound[channel] = optimized_spend_unbounded[channel]
            optm_response_unit_unbound[channel] = optimized_responses_unbounded[channel]
            optm_response_marg_unit_unbound[channel] = (
                self._fx_objective(
                    x=optimized_spend_unbounded[channel] + last_allocation_unb,
                    coeff=self.coefs_eval[channel],
                    alpha=self.alphas_eval[f"{channel}_alphas"],
                    inflexion=self.inflexions_eval[f"{channel}_gammas"],
                    x_hist_carryover=x_hist_carryover[channel],
                    theta=self.thetas[channel],
                    get_sum=True,
                )
                - optimized_responses_unbounded[channel]
            ) / last_allocation_unb

        # Handle skipped channels
        channels_to_drop = np.isin(
            self.media_spend_sorted,
            np.concatenate([self.zero_coef_channel, self.zero_constraint_channel]),
        )
        for channel in self.media_spend_sorted[channels_to_drop]:
            for result_dict in [
                optm_spend_unit,
                optm_response_unit,
                optm_response_marg_unit,
                optm_spend_unit_unbound,
                optm_response_unit_unbound,
                optm_response_marg_unit_unbound,
            ]:
                result_dict[channel] = 0

        # Build results dictionary
        optimized_results = {
            "nls_mod": {
                "solution": optm_spend_unit.values,
                "objective": -sum(optimized_responses.values()),
                "status": 5,
            },
            "nls_mod_unbound": {
                "solution": optm_spend_unit_unbound.values,
                "objective": -sum(optimized_responses_unbounded.values()),
                "status": 5,
            },
            "optm_spend_unit": optm_spend_unit,
            "optm_spend_unit_unbound": optm_spend_unit_unbound,
            "optm_response_unit": optm_response_unit,
            "optm_response_unit_unbound": optm_response_unit_unbound,
            "optm_response_marg_unit": optm_response_marg_unit,
            "optm_response_marg_unit_unbound": optm_response_marg_unit_unbound,
            "channels_mapped": np.ones(len(self.media_spend_sorted), dtype=bool),
        }

        dt_optim_out = self._build_optim_output(optimized_results)

        # Log comparison
        total_init_response = float(sum(original_spend.values))
        total_opt_response = float(sum(optimized_responses.values()))
        response_lift = float((total_opt_response / total_init_response - 1) * 100)

        self.logger.debug("Iterative allocation completed")
        self.logger.debug(f"Total initial response: {total_init_response}")
        self.logger.debug(f"Total optimized response: {total_opt_response}")
        self.logger.debug(f"Response lift: {response_lift:.2f}%")

        # Create comparison dataframe
        comparison = pd.DataFrame(
            {
                "Channel": self.channel_for_allocation,
                "Initial Spend": [
                    original_spend[c] for c in self.channel_for_allocation
                ],
                "Optimized Spend": [
                    optimized_spend[c] for c in self.channel_for_allocation
                ],
                "Spend Change %": [
                    (optimized_spend[c] / original_spend[c] - 1) * 100
                    for c in self.channel_for_allocation
                ],
            }
        )

        print("Budget allocation results:")
        print(comparison)

        return dt_optim_out

    def _build_optim_output(self, optim_results):
        """Build optimization output DataFrame matching R's structure"""

        # Get unique simulation period
        unique_sim_period = self.simulation_period.iloc[0]

        # Create output DataFrame
        dt_optim_out = pd.DataFrame(
            {
                "sol_id": self.select_model,
                "dep_var_type": self.mmm_data.mmmdata_spec.dep_var_type,
                "channels": self.media_spend_sorted,
                "date_min": self.date_min,
                "date_max": self.date_max,
                "periods": f"{unique_sim_period} {self.mmm_data.mmmdata_spec.interval_type}s",
                # Constraints
                "constr_low": self.temp_lb_all,
                "constr_low_abs": self.lb_all,
                "constr_up": self.temp_ub_all,
                "constr_up_abs": self.ub_all,
                "unconstr_mult": self.params.channel_constr_multiplier,
                "constr_low_unb": self.temp_lb_ext_all,
                "constr_low_unb_abs": self.lb_ext_all,
                "constr_up_unb": self.temp_ub_ext_all,
                "constr_up_unb_abs": self.ub_ext_all,
                # Historical spends
                "histSpendAll": self.hist_spend_all,
                "histSpendAllTotal": self.hist_spend_all_total,
                "histSpendAllUnit": self.hist_spend_all_unit,
                "histSpendAllUnitTotal": self.hist_spend_all_unit_total,
                "histSpendAllShare": self.hist_spend_all_share,
                "histSpendWindow": self.hist_spend_window,
                "histSpendWindowTotal": self.hist_spend_window_total,
                "histSpendWindowUnit": self.hist_spend_window_unit,
                "histSpendWindowUnitTotal": self.hist_spend_window_unit_total,
                "histSpendWindowShare": self.hist_spend_window_share,
                # Initial spends
                "initSpendUnit": self.init_spend_unit,
                "initSpendUnitTotal": self.init_spend_unit_total,
                "initSpendShare": self.init_spend_share,
                "initSpendTotal": self.init_spend_unit_total * unique_sim_period,
                # Initial responses
                "initResponseUnit": self.init_response_unit,
                "initResponseUnitTotal": self.init_response_unit.sum(),
                "initResponseMargUnit": self.init_response_marg_unit,
                "initResponseTotal": self.init_response_unit.sum() * unique_sim_period,
                "initResponseUnitShare": self.init_response_unit
                / self.init_response_unit.sum(),
                "initRoiUnit": self.init_response_unit / self.init_spend_unit,
                "initCpaUnit": self.init_spend_unit / self.init_response_unit,
                # Budget
                "total_budget_unit": self.total_budget_unit,
                "total_budget_unit_delta": self.total_budget_unit
                / self.init_spend_unit_total
                - 1,
                # Optimized results - bounded
                "optmSpendUnit": optim_results["optm_spend_unit"],
                "optmSpendUnitDelta": (
                    optim_results["optm_spend_unit"] / self.init_spend_unit - 1
                ),
                "optmSpendUnitTotal": optim_results["optm_spend_unit"].sum(),
                "optmSpendUnitTotalDelta": optim_results["optm_spend_unit"].sum()
                / self.init_spend_unit_total
                - 1,
                "optmSpendShareUnit": optim_results["optm_spend_unit"]
                / optim_results["optm_spend_unit"].sum(),
                "optmSpendTotal": optim_results["optm_spend_unit"].sum()
                * unique_sim_period,
                # Optimized results - unbounded
                "optmSpendUnitUnbound": optim_results["optm_spend_unit_unbound"],
                "optmSpendUnitDeltaUnbound": (
                    optim_results["optm_spend_unit_unbound"] / self.init_spend_unit - 1
                ),
                "optmSpendUnitTotalUnbound": optim_results[
                    "optm_spend_unit_unbound"
                ].sum(),
                "optmSpendUnitTotalDeltaUnbound": optim_results[
                    "optm_spend_unit_unbound"
                ].sum()
                / self.init_spend_unit_total
                - 1,
                "optmSpendShareUnitUnbound": optim_results["optm_spend_unit_unbound"]
                / optim_results["optm_spend_unit_unbound"].sum(),
                "optmSpendTotalUnbound": optim_results["optm_spend_unit_unbound"].sum()
                * unique_sim_period,
                # Response metrics - bounded
                "optmResponseUnit": optim_results["optm_response_unit"],
                "optmResponseMargUnit": optim_results["optm_response_marg_unit"],
                "optmResponseUnitTotal": optim_results["optm_response_unit"].sum(),
                "optmResponseTotal": optim_results["optm_response_unit"].sum()
                * unique_sim_period,
                "optmResponseUnitShare": optim_results["optm_response_unit"]
                / optim_results["optm_response_unit"].sum(),
                "optmRoiUnit": optim_results["optm_response_unit"]
                / optim_results["optm_spend_unit"],
                "optmCpaUnit": optim_results["optm_spend_unit"]
                / optim_results["optm_response_unit"],
                "optmResponseUnitLift": (
                    optim_results["optm_response_unit"] / self.init_response_unit
                )
                - 1,
                # Response metrics - unbounded
                "optmResponseUnitUnbound": optim_results["optm_response_unit_unbound"],
                "optmResponseMargUnitUnbound": optim_results[
                    "optm_response_marg_unit_unbound"
                ],
                "optmResponseUnitTotalUnbound": optim_results[
                    "optm_response_unit_unbound"
                ].sum(),
                "optmResponseTotalUnbound": optim_results[
                    "optm_response_unit_unbound"
                ].sum()
                * unique_sim_period,
                "optmResponseUnitShareUnbound": optim_results[
                    "optm_response_unit_unbound"
                ]
                / optim_results["optm_response_unit_unbound"].sum(),
                "optmRoiUnitUnbound": optim_results["optm_response_unit_unbound"]
                / optim_results["optm_spend_unit_unbound"],
                "optmCpaUnitUnbound": optim_results["optm_spend_unit_unbound"]
                / optim_results["optm_response_unit_unbound"],
                "optmResponseUnitLiftUnbound": (
                    optim_results["optm_response_unit_unbound"]
                    / self.init_response_unit
                )
                - 1,
            }
        )

        # Add calculated fields
        dt_optim_out["optmResponseUnitTotalLift"] = (
            dt_optim_out["optmResponseUnitTotal"]
            / dt_optim_out["initResponseUnitTotal"]
        ) - 1
        dt_optim_out["optmResponseUnitTotalLiftUnbound"] = (
            dt_optim_out["optmResponseUnitTotalUnbound"]
            / dt_optim_out["initResponseUnitTotal"]
        ) - 1

        return dt_optim_out

    def _calculate_curves(self, dt_optim_out: pd.DataFrame) -> Dict:
        """Calculate curves and main points for each channel."""

        # Set level names based on scenario
        if self.params.scenario == "max_response":
            levs1 = [
                "Initial",
                "Bounded",
                f"Bounded x{self.params.channel_constr_multiplier}",
            ]
        elif self.params.scenario == "target_efficiency":
            if self.mmm_data.mmmdata_spec.dep_var_type == "revenue":
                levs1 = [
                    "Initial",
                    f"Hit ROAS {round(self.params.target_value, 2)}",
                    f"Hit ROAS {self.target_value_ext}",
                ]
            else:
                levs1 = [
                    "Initial",
                    f"Hit CPA {round(self.params.target_value, 2)}",
                    f"Hit CPA {round(self.target_value_ext, 2)}",
                ]

        # Create DataFrame for S-curve
        dt_optim_out_scurve = pd.concat(
            [
                # Initial points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["initSpendUnit"],
                        "response": dt_optim_out["initResponseUnit"],
                        "type": levs1[0],
                    }
                ),
                # Bounded optimization points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["optmSpendUnit"],
                        "response": dt_optim_out["optmResponseUnit"],
                        "type": levs1[1],
                    }
                ),
                # Unbounded optimization points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": dt_optim_out["optmSpendUnitUnbound"],
                        "response": dt_optim_out["optmResponseUnitUnbound"],
                        "type": levs1[2],
                    }
                ),
                # Carryover points
                pd.DataFrame(
                    {
                        "channels": dt_optim_out["channels"],
                        "spend": 0,
                        "response": 0,
                        "type": "Carryover",
                    }
                ),
            ]
        ).assign(
            spend=lambda x: pd.to_numeric(x["spend"]),
            response=lambda x: pd.to_numeric(x["response"]),
        )  # Remove the groupby here

        return {"dt_optim_out_scurve": dt_optim_out_scurve, "levs1": levs1}

    def _prepare_scurve_data(
        self, dt_optim_out_scurve: pd.DataFrame, levs1: List[str]
    ) -> Dict:
        """Prepare S-curve data for plotting."""

        plot_dt_scurve = []

        # Process each channel
        for channel in self.channel_for_allocation:
            # Get carryover vector for this channel
            carryover_vec = self.hist_carryover_eval[channel]
            carryover_mean = np.mean(carryover_vec)

            # Update spend values with carryover
            mask_levs = dt_optim_out_scurve["type"].isin(levs1) & (
                dt_optim_out_scurve["channels"] == channel
            )
            mask_carryover = (dt_optim_out_scurve["type"] == "Carryover") & (
                dt_optim_out_scurve["channels"] == channel
            )

            dt_optim_out_scurve.loc[mask_levs, "spend"] += carryover_mean
            dt_optim_out_scurve.loc[mask_carryover, "spend"] = carryover_mean

            # Generate simulation points
            max_x = max(
                dt_optim_out_scurve[dt_optim_out_scurve["channels"] == channel][
                    "spend"
                ].max()
                * 10,
                50_000_000,
            )
            simulate_spend = np.linspace(0, max_x, 2000)

            # Choose carryover value based on flag
            carryover_for_simulation = (
                carryover_mean if self.params.include_scurve_carryover else 0
            )

            simulate_response = self._fx_objective(
                x=simulate_spend,
                coeff=self.coefs_eval[channel],
                alpha=self.alphas_eval[f"{channel}_alphas"],
                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=carryover_for_simulation,
                theta=self.thetas[channel],  # Add theta parameter
                get_sum=False,
            )

            simulate_response_carryover = self._fx_objective(
                x=carryover_mean,
                coeff=self.coefs_eval[channel],
                alpha=self.alphas_eval[f"{channel}_alphas"],
                inflexion=self.inflexions_eval[f"{channel}_gammas"],
                x_hist_carryover=0,
                theta=self.thetas[channel],  # Add theta parameter
                get_sum=True,
            )

            # Create arrays of consistent length
            n = len(simulate_spend)
            channel_array = [channel] * n
            mean_carryover_array = [carryover_mean] * n
            carryover_response_array = [simulate_response_carryover] * n

            # Handle the case where simulate_response might be a scalar instead of an array
            if np.isscalar(simulate_response) or len(simulate_response) != n:
                total_response_array = [simulate_response] * n
            else:
                total_response_array = simulate_response

            # Store simulation data with consistent array lengths
            plot_dt_scurve.append(
                pd.DataFrame(
                    {
                        "channel": channel_array,
                        "spend": simulate_spend,
                        "mean_carryover": mean_carryover_array,
                        "carryover_response": carryover_response_array,
                        "total_response": total_response_array,
                    }
                )
            )

            # Update carryover response in main dataframe
            dt_optim_out_scurve.loc[mask_carryover, "response"] = (
                simulate_response_carryover
            )

        # Combine all simulation data
        plot_dt_scurve = pd.concat(plot_dt_scurve, ignore_index=True)

        # Prepare main points data
        main_points = dt_optim_out_scurve.rename(
            columns={
                "response": "response_point",
                "spend": "spend_point",
                "channels": "channel",
            }
        )

        # Log main_points with more information
        self.logger.debug(
            json.dumps(
                {
                    "step": "main_points",
                    "shape": main_points.shape,
                    "columns": list(main_points.columns),
                    "data": main_points.to_dict(
                        orient="records"
                    ),  # 'records' is usually more readable than 'index'
                    "types": {
                        col: str(main_points[col].dtype) for col in main_points.columns
                    },
                },
                indent=2,
                default=str,  # Handles any non-serializable objects like timestamps
            )
        )
        # Calculate mean spend
        temp_caov = main_points[main_points["type"] == "Carryover"]

        # Create a mapping of channel to carryover spend
        caov_spends = dict(zip(temp_caov["channel"], temp_caov["spend_point"]))

        # Initialize mean_spend with spend_point
        main_points["mean_spend"] = main_points["spend_point"]

        # For non-carryover rows, subtract the corresponding channel's carryover spend
        for channel in caov_spends:
            mask = (main_points["channel"] == channel) & (
                main_points["type"] != "Carryover"
            )
            main_points.loc[mask, "mean_spend"] -= caov_spends[channel]

        # Handle duplicate level names
        if levs1[1] == levs1[2]:
            levs1[2] = f"{levs1[2]}."

        # Set factor levels for type
        main_points["type"] = pd.Categorical(
            main_points["type"], categories=["Carryover"] + levs1, ordered=True
        )

        # Calculate ROI and response metrics
        main_points["roi_mean"] = (
            main_points["response_point"] / main_points["mean_spend"]
        )

        # Calculate marginal responses
        mresp_caov = main_points[main_points["type"] == "Carryover"][
            "response_point"
        ].values
        mresp_init = (
            main_points[main_points["type"] == levs1[0]]["response_point"].values
            - mresp_caov
        )
        mresp_b = (
            main_points[main_points["type"] == levs1[1]]["response_point"].values
            - mresp_caov
        )
        mresp_unb = (
            main_points[main_points["type"] == levs1[2]]["response_point"].values
            - mresp_caov
        )

        main_points["marginal_response"] = np.concatenate(
            [
                mresp_init,
                mresp_b,
                mresp_unb,
                np.zeros(len(mresp_init)),  # For Carryover points
            ]
        )

        # Calculate marginal metrics
        main_points["roi_marginal"] = (
            main_points["marginal_response"] / main_points["mean_spend"]
        )
        main_points["cpa_marginal"] = (
            main_points["mean_spend"] / main_points["marginal_response"]
        )
        self.eval_dict["mainPoints"] = main_points
        self.eval_dict["plotDT_scurve"] = plot_dt_scurve
        return {"plot_dt_scurve": plot_dt_scurve, "main_points": main_points}
