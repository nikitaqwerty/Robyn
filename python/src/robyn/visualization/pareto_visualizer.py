from pathlib import Path
import re
from typing import Dict, List, Optional, Union
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from robyn.modeling.entities.modeloutputs import ModelOutputs
import seaborn as sns
import logging
from robyn.data.entities.enums import ProphetVariableType
from robyn.data.entities.holidays_data import HolidaysData
from robyn.modeling.entities.featurized_mmm_data import FeaturizedMMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.data.entities.hyperparameters import AdstockType, Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.visualization.base_visualizer import BaseVisualizer
from robyn.data.entities.enums import DependentVarType
import math
import matplotlib.dates as mdates

logger = logging.getLogger(__name__)


class ParetoVisualizer(BaseVisualizer):
    def __init__(
        self,
        pareto_result: ParetoResult,
        mmm_data: MMMData,
        holiday_data: Optional[HolidaysData] = None,
        hyperparameter: Optional[Hyperparameters] = None,
        featurized_mmm_data: Optional[FeaturizedMMMData] = None,
        unfiltered_pareto_result: Optional[ParetoResult] = None,
        model_outputs: Optional[ModelOutputs] = None,
    ):
        super().__init__()
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data
        self.holiday_data = holiday_data
        self.hyperparameter = hyperparameter
        self.featurized_mmm_data = featurized_mmm_data
        self.unfiltered_pareto_result = unfiltered_pareto_result
        self.model_outputs = model_outputs

    def _baseline_vars(
        self, baseline_level, prophet_vars: List[ProphetVariableType] = []
    ) -> list:
        """
        Returns a list of baseline variables based on the provided level.
        Args:
            InputCollect (dict): A dictionary containing various input data.
            baseline_level (int): The level of baseline variables to include.
        Returns:
            list: A list of baseline variable names.
        """
        # Check if baseline_level is valid
        if baseline_level < 0 or baseline_level > 5:
            raise ValueError("baseline_level must be an integer between 0 and 5")
        baseline_variables = []
        # Level 1: Include intercept variables
        if baseline_level >= 1:
            baseline_variables.extend(["(Intercept)", "intercept"])
        # Level 2: Include trend variables
        if baseline_level >= 2:
            baseline_variables.append("trend")
        # Level 3: Include prophet variables
        if baseline_level >= 3:
            baseline_variables.extend(list(set(baseline_variables + prophet_vars)))
        # Level 4: Include context variables
        if baseline_level >= 4:
            baseline_variables.extend(self.mmm_data.mmmdata_spec.context_vars)
        # Level 5: Include organic variables
        if baseline_level >= 5:
            baseline_variables.extend(self.mmm_data.mmmdata_spec.organic_vars)
        return list(set(baseline_variables))

    def format_number(self, x: float, pos=None) -> str:
        """Format large numbers with K/M/B abbreviations.

        Args:
            x: Number to format
            pos: Position (required by matplotlib FuncFormatter but not used)

        Returns:
            Formatted string
        """
        if abs(x) >= 1e9:
            return f"{x/1e9:.1f}B"
        elif abs(x) >= 1e6:
            return f"{x/1e6:.1f}M"
        elif abs(x) >= 1e3:
            return f"{x/1e3:.1f}K"
        else:
            return f"{x:.1f}"

    def generate_waterfall(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        baseline_level: int = 0,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate waterfall chart for specific solution."""

        logger.debug("Starting generation of waterfall plot")
        if solution_id not in self.pareto_result.plot_data_collect:
            # Check if this might be a solution from unfiltered results that wasn't processed
            if (
                self.unfiltered_pareto_result
                and solution_id
                in self.unfiltered_pareto_result.result_hyp_param["sol_id"].values
            ):
                logger.warning(
                    f"Solution ID {solution_id} found in unfiltered results but not in plot_data_collect. "
                    "This solution's plot data was not generated. Use ParetoOptimizer.optimize() with the "
                    "fix that includes best solutions from unfiltered results."
                )
            else:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in any available data."
                )
            return None

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        # Avoid copying if modifications are local to this function
        waterfall_data = plot_data["plot2data"]["plotWaterfallLoop"]

        # Get baseline variables
        prophet_vars = self.holiday_data.prophet_vars if self.holiday_data else []
        baseline_vars = self._baseline_vars(baseline_level, prophet_vars)

        # Transform baseline variables
        waterfall_data["rn"] = np.where(
            waterfall_data["rn"].isin(baseline_vars),
            f"Baseline_L{baseline_level}",
            waterfall_data["rn"],
        )

        # Group and summarize
        waterfall_data = (
            waterfall_data.groupby("rn", as_index=False)
            .agg({"xDecompAgg": "sum", "xDecompPerc": "sum"})
            .reset_index()
        )

        # Sort by percentage contribution
        waterfall_data = waterfall_data.sort_values("xDecompPerc", ascending=True)

        # Calculate waterfall positions
        waterfall_data["end"] = 1 - waterfall_data["xDecompPerc"].cumsum()
        waterfall_data["start"] = waterfall_data["end"].shift(1)
        waterfall_data["start"] = waterfall_data["start"].fillna(1)
        waterfall_data["sign"] = np.where(
            waterfall_data["xDecompPerc"] >= 0, "Positive", "Negative"
        )

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))
        else:
            fig = None

        # Define colors
        colors = {"Positive": "#59B3D2", "Negative": "#E5586E"}

        # Create categorical y-axis positions
        y_pos = np.arange(len(waterfall_data))

        # Create horizontal bars
        bars = ax.barh(
            y=y_pos,
            width=waterfall_data["start"] - waterfall_data["end"],
            left=waterfall_data["end"],
            color=[colors[sign] for sign in waterfall_data["sign"]],
            height=0.6,
        )

        # Add text labels
        for idx, row in enumerate(waterfall_data.itertuples()):
            # Format label text
            if abs(row.xDecompAgg) >= 1e9:
                formatted_num = f"{row.xDecompAgg/1e9:.1f}B"
            elif abs(row.xDecompAgg) >= 1e6:
                formatted_num = f"{row.xDecompAgg/1e6:.1f}M"
            elif abs(row.xDecompAgg) >= 1e3:
                formatted_num = f"{row.xDecompAgg/1e3:.1f}K"
            else:
                formatted_num = f"{row.xDecompAgg:.1f}"

            # Calculate x-position as the middle of the bar
            x_pos = (row.start + row.end) / 2

            # Use y_pos[idx] to ensure alignment with bars
            ax.text(
                x_pos,
                y_pos[idx],  # Use the same y-position as the corresponding bar
                f"{formatted_num}\n{row.xDecompPerc*100:.1f}%",
                ha="center",  # Center align horizontally
                va="center",  # Center align vertically
                fontsize=9,
                linespacing=0.9,
            )

        # Set y-ticks and labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(waterfall_data["rn"])

        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0%}".format(x)))
        ax.set_xticks(np.arange(0, 1.1, 0.2))

        # Set plot limits
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.5, len(waterfall_data) - 0.5)

        # Add legend at top
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor=colors["Positive"], label="Positive"),
            Patch(facecolor=colors["Negative"], label="Negative"),
        ]

        # Create legend with white background
        legend = ax.legend(
            handles=legend_elements,
            title="Sign",
            loc="upper left",
            bbox_to_anchor=(0, 1.15),
            ncol=2,
            frameon=True,
            framealpha=1.0,
        )

        # Set title
        ax.set_title("Response Decomposition Waterfall", pad=30, x=0.5, y=1.05)

        # Label axes
        ax.set_xlabel("Contribution")
        ax.set_ylabel(None)

        # Customize grid
        ax.grid(True, axis="x", alpha=0.2)
        ax.set_axisbelow(True)

        logger.debug("Successfully generated waterfall plot")
        # Adjust layout
        if fig:
            plt.subplots_adjust(right=0.85, top=0.85)
            fig = plt.gcf()
            # Add metrics text if available
            if metrics:
                metrics_to_display = {
                    k: metrics.get(k, float("nan"))  # Use NaN for missing metrics
                    for k in [
                        "rsq_train",
                        "rsq_val",
                        "rsq_test",
                        "nrmse",
                        "nrmse_train",
                        "nrmse_val",
                        "nrmse_test",
                        "decomp.rssd",
                    ]
                }
                metrics_str_lines = [
                    f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
                    f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
                    f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
                ]
                # Place text below the title
                fig.text(
                    0.5,
                    0.95,
                    f"Metrics for Solution {solution_id}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                )
                fig.text(
                    0.5,
                    0.93,
                    metrics_str_lines[0],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.91,
                    metrics_str_lines[1],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.89,
                    metrics_str_lines[2],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                # Adjust top margin to make space for the text
                plt.subplots_adjust(top=0.84)  # Reduced top margin further

            plt.close(fig)
            return fig

        return None

    def generate_fitted_vs_actual(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate time series plot comparing fitted vs actual values.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object
        """

        logger.debug("Starting generation of fitted vs actual plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            # Check if this might be a solution from unfiltered results that wasn't processed
            if (
                self.unfiltered_pareto_result
                and solution_id
                in self.unfiltered_pareto_result.result_hyp_param["sol_id"].values
            ):
                logger.warning(
                    f"Solution ID {solution_id} found in unfiltered results but not in plot_data_collect. "
                    "This solution's plot data was not generated. Use ParetoOptimizer.optimize() with the "
                    "fix that includes best solutions from unfiltered results."
                )
            else:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in any available data."
                )
            return None

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        # Avoid copying if modifications are local to this function
        ts_data = plot_data["plot5data"]["xDecompVecPlotMelted"]

        # Ensure ds column is datetime and remove any NaT values
        ts_data["ds"] = pd.to_datetime(ts_data["ds"])
        ts_data = ts_data.dropna(subset=["ds"])  # Remove rows with NaT dates

        if ts_data.empty:
            logger.warning(f"No valid date data found for solution {solution_id}")
            return None

        ts_data["linetype"] = np.where(
            ts_data["variable"] == "predicted", "solid", "dotted"
        )
        ts_data["variable"] = ts_data["variable"].str.title()

        # Get train_size from x_decomp_agg
        train_size_series = self.pareto_result.x_decomp_agg[
            self.pareto_result.x_decomp_agg["sol_id"] == solution_id
        ]["train_size"]

        if not train_size_series.empty:
            train_size = float(train_size_series.iloc[0])
        else:
            train_size = 0

        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        else:
            fig = None

        colors = {
            "Actual": "#FF6B00",  # Darker orange
            "Predicted": "#0066CC",  # Darker blue
        }

        # Plot lines with different styles for predicted vs actual
        for var in ts_data["variable"].unique():
            var_data = ts_data[ts_data["variable"] == var]
            linestyle = "solid" if var_data["linetype"].iloc[0] == "solid" else "dotted"
            ax.plot(
                var_data["ds"],
                var_data["value"],
                label=var,
                linestyle=linestyle,
                linewidth=1,
                color=colors[var],
            )

        # Format y-axis with abbreviations
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

        # Set y-axis limits with some padding
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, y_max * 1.2)  # Add 20% padding at the top

        # Add training/validation/test splits if train_size exists and is valid
        if train_size > 0:
            try:
                # Get unique sorted dates, excluding NaT
                unique_dates = sorted(ts_data["ds"].dropna().unique())
                total_days = len(unique_dates)

                if total_days > 0:
                    # Calculate split points
                    train_cut = int(total_days * train_size)
                    val_cut = train_cut + int(total_days * (1 - train_size) / 2)

                    # Get dates for splits
                    splits = [
                        (train_cut, "Train", train_size),
                        (val_cut, "Validation", (1 - train_size) / 2),
                        (total_days - 1, "Test", (1 - train_size) / 2),
                    ]

                    # Get y-axis limits for text placement
                    y_min, y_max = ax.get_ylim()

                    # Add vertical lines and labels
                    for idx, label, size in splits:
                        if 0 <= idx < len(unique_dates):  # Ensure index is valid
                            date = unique_dates[idx]
                            if pd.notna(date):  # Check if date is valid
                                # Add vertical line - extend beyond the top of the plot
                                ax.axvline(
                                    date, color="#39638b", alpha=0.8, ymin=0, ymax=1.1
                                )

                                # Add rotated text label
                                ax.text(
                                    date,
                                    y_max,
                                    f"{label}: {size*100:.1f}%",
                                    rotation=270,
                                    color="#39638b",
                                    alpha=0.5,
                                    size=9,
                                    ha="left",
                                    va="top",
                                )
            except Exception as e:
                logger.warning(f"Error adding split lines: {str(e)}")
                # Continue with the rest of the plot even if split lines fail

        # Set title and labels
        ax.set_title("Actual vs. Predicted Response", pad=20)
        ax.set_xlabel("Date")
        ax.set_ylabel("Response")

        # Configure legend
        ax.legend(
            bbox_to_anchor=(0.01, 1.02),  # Position at top-left
            loc="lower left",
            ncol=2,  # Two columns side by side
            borderaxespad=0,
            frameon=False,
            fontsize=7,
            handlelength=2,  # Length of the legend lines
            handletextpad=0.5,  # Space between line and text
            columnspacing=1.0,  # Space between columns
        )

        # Grid styling
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

        # Format dates on x-axis using datetime locator and formatter
        years = mdates.YearLocator()
        years_fmt = mdates.DateFormatter("%Y")
        ax.xaxis.set_major_locator(years)
        ax.xaxis.set_major_formatter(years_fmt)

        logger.debug("Successfully generated fitted vs actual plot")
        if fig:
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            fig = plt.gcf()
            # Add metrics text if available
            if metrics:
                metrics_to_display = {
                    k: metrics.get(k, float("nan"))  # Use NaN for missing metrics
                    for k in [
                        "rsq_train",
                        "rsq_val",
                        "rsq_test",
                        "nrmse",
                        "nrmse_train",
                        "nrmse_val",
                        "nrmse_test",
                        "decomp.rssd",
                    ]
                }
                metrics_str_lines = [
                    f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
                    f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
                    f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
                ]
                # Place text below the title
                fig.text(
                    0.5,
                    0.95,
                    f"Metrics for Solution {solution_id}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                )
                fig.text(
                    0.5,
                    0.93,
                    metrics_str_lines[0],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.91,
                    metrics_str_lines[1],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.89,
                    metrics_str_lines[2],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                # Adjust top margin to make space for the text
                plt.subplots_adjust(top=0.84)  # Adjust top margin

            plt.close(fig)
            return fig
        return None

    def generate_diagnostic_plot(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate diagnostic scatter plot of fitted vs residual values.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object
        """

        logger.debug("Starting generation of diagnostic plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            # Check if this might be a solution from unfiltered results that wasn't processed
            if (
                self.unfiltered_pareto_result
                and solution_id
                in self.unfiltered_pareto_result.result_hyp_param["sol_id"].values
            ):
                logger.warning(
                    f"Solution ID {solution_id} found in unfiltered results but not in plot_data_collect. "
                    "This solution's plot data was not generated. Use ParetoOptimizer.optimize() with the "
                    "fix that includes best solutions from unfiltered results."
                )
            else:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in any available data."
                )
            return None

        # Get data for specific solution
        plot_data = self.pareto_result.plot_data_collect[solution_id]
        diag_data = plot_data["plot6data"]["xDecompVecPlot"]

        # Calculate residuals
        diag_data["residuals"] = diag_data["actual"] - diag_data["predicted"]

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        # Create scatter plot
        ax.scatter(
            diag_data["predicted"], diag_data["residuals"], alpha=0.5, color="steelblue"
        )

        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        # Add smoothed line with confidence interval
        from scipy.stats import gaussian_kde

        x_smooth = np.linspace(
            diag_data["predicted"].min(), diag_data["predicted"].max(), 100
        )

        # Fit LOWESS
        from statsmodels.nonparametric.smoothers_lowess import lowess

        smoothed = lowess(diag_data["residuals"], diag_data["predicted"], frac=0.2)

        # Plot smoothed line
        ax.plot(smoothed[:, 0], smoothed[:, 1], color="red", linewidth=2, alpha=0.8)

        # Calculate confidence intervals (using standard error bands)
        residual_std = np.std(diag_data["residuals"])
        ax.fill_between(
            smoothed[:, 0],
            smoothed[:, 1] - 2 * residual_std,
            smoothed[:, 1] + 2 * residual_std,
            color="red",
            alpha=0.1,
        )

        # Format axes with abbreviations
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(self.format_number))

        # Set labels and title
        ax.set_xlabel("Fitted")
        ax.set_ylabel("Residual")
        ax.set_title("Fitted vs. Residual")

        # Customize grid
        ax.grid(True, alpha=0.2)
        ax.set_axisbelow(True)

        # Use white background
        ax.set_facecolor("white")

        logger.debug("Successfully generated of diagnostic plot")

        if fig:
            plt.tight_layout()
            fig = plt.gcf()
            # Add metrics text if available
            if metrics:
                metrics_to_display = {
                    k: metrics.get(k, float("nan"))  # Use NaN for missing metrics
                    for k in [
                        "rsq_train",
                        "rsq_val",
                        "rsq_test",
                        "nrmse",
                        "nrmse_train",
                        "nrmse_val",
                        "nrmse_test",
                        "decomp.rssd",
                    ]
                }
                metrics_str_lines = [
                    f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
                    f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
                    f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
                ]
                # Place text below the title
                fig.text(
                    0.5,
                    0.95,
                    f"Metrics for Solution {solution_id}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                )
                fig.text(
                    0.5,
                    0.93,
                    metrics_str_lines[0],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.91,
                    metrics_str_lines[1],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.89,
                    metrics_str_lines[2],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                # Adjust top margin to make space for the text
                plt.subplots_adjust(top=0.84)  # Adjust top margin

            plt.close(fig)
            return fig
        return None

    def generate_immediate_vs_carryover(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate stacked bar chart comparing immediate vs carryover effects.

        Args:
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            plt.Figure if ax is None, else None
        """

        logger.debug("Starting generation of immediate vs carryover plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            # Check if this might be a solution from unfiltered results that wasn't processed
            if (
                self.unfiltered_pareto_result
                and solution_id
                in self.unfiltered_pareto_result.result_hyp_param["sol_id"].values
            ):
                logger.warning(
                    f"Solution ID {solution_id} found in unfiltered results but not in plot_data_collect. "
                    "This solution's plot data was not generated. Use ParetoOptimizer.optimize() with the "
                    "fix that includes best solutions from unfiltered results."
                )
            else:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in any available data."
                )
            return None

        plot_data = self.pareto_result.plot_data_collect[solution_id]
        df_imme_caov = plot_data["plot7data"]

        # Ensure percentage is numeric
        df_imme_caov["percentage"] = pd.to_numeric(
            df_imme_caov["percentage"], errors="coerce"
        )

        # Sort channels alphabetically
        df_imme_caov = df_imme_caov.sort_values("rn", ascending=True)

        # Set up type factor levels matching R plot order
        df_imme_caov["type"] = pd.Categorical(
            df_imme_caov["type"], categories=["Immediate", "Carryover"], ordered=True
        )

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        colors = {"Immediate": "#59B3D2", "Carryover": "coral"}

        bottom = np.zeros(len(df_imme_caov["rn"].unique()))
        y_pos = range(len(df_imme_caov["rn"].unique()))
        channels = df_imme_caov["rn"].unique()
        types = ["Immediate", "Carryover"]  # Order changed to Immediate first

        # Normalize percentages to sum to 100% for each channel
        for channel in channels:
            mask = df_imme_caov["rn"] == channel
            total = df_imme_caov.loc[mask, "percentage"].sum()
            if total > 0:  # Avoid division by zero
                df_imme_caov.loc[mask, "percentage"] = (
                    df_imme_caov.loc[mask, "percentage"] / total
                )

        for type_name in types:
            type_data = df_imme_caov[df_imme_caov["type"] == type_name]
            percentages = type_data["percentage"].values

            bars = ax.barh(
                y_pos,
                percentages,
                left=bottom,
                height=0.5,
                label=type_name,
                color=colors[type_name],
            )

            for i, (rect, percentage) in enumerate(zip(bars, percentages)):
                width = rect.get_width()
                x_pos = bottom[i] + width / 2
                try:
                    percentage_text = f"{round(float(percentage) * 100)}%"
                except (ValueError, TypeError):
                    percentage_text = "0%"
                ax.text(x_pos, i, percentage_text, ha="center", va="center")

            bottom += percentages

        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
        ax.set_xlim(0, 1)

        # Reduced legend size
        ax.legend(
            title=None,
            bbox_to_anchor=(0, 1.02, 0.15, 0.1),  # Reduced width from 0.3 to 0.2
            loc="lower left",
            ncol=2,
            mode="expand",
            borderaxespad=0,
            frameon=False,
            fontsize=7,  # Reduced from 8 to 7
        )

        ax.set_xlabel("% Response")
        ax.set_ylabel(None)
        ax.set_title("Immediate vs. Carryover Response Percentage", pad=50, y=1.2)

        ax.grid(True, axis="x", alpha=0.2)
        ax.grid(False, axis="y")
        ax.set_axisbelow(True)
        ax.set_facecolor("white")

        if fig:
            plt.tight_layout()
            plt.subplots_adjust(top=0.85)
            fig = plt.gcf()
            # Add metrics text if available
            if metrics:
                metrics_to_display = {
                    k: metrics.get(k, float("nan"))  # Use NaN for missing metrics
                    for k in [
                        "rsq_train",
                        "rsq_val",
                        "rsq_test",
                        "nrmse",
                        "nrmse_train",
                        "nrmse_val",
                        "nrmse_test",
                        "decomp.rssd",
                    ]
                }
                metrics_str_lines = [
                    f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
                    f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
                    f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
                ]
                # Place text below the title
                fig.text(
                    0.5,
                    0.95,
                    f"Metrics for Solution {solution_id}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                )
                fig.text(
                    0.5,
                    0.93,
                    metrics_str_lines[0],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.91,
                    metrics_str_lines[1],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.89,
                    metrics_str_lines[2],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                # Adjust top margin to make space for the text
                plt.subplots_adjust(top=0.84)  # Adjust top margin

            plt.close(fig)
            return fig
        return None

    def generate_adstock_rate(
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate adstock rate visualization based on adstock type.

        Args:
            solution_id: ID of solution to visualize
            ax: Optional matplotlib axes to plot on. If None, creates new figure

        Returns:
            Optional[plt.Figure]: Generated figure if ax is None, otherwise None
        """

        logger.debug("Starting generation of adstock plot")

        if solution_id not in self.pareto_result.plot_data_collect:
            # Check if this might be a solution from unfiltered results that wasn't processed
            if (
                self.unfiltered_pareto_result
                and solution_id
                in self.unfiltered_pareto_result.result_hyp_param["sol_id"].values
            ):
                logger.warning(
                    f"Solution ID {solution_id} found in unfiltered results but not in plot_data_collect. "
                    "This solution's plot data was not generated. Use ParetoOptimizer.optimize() with the "
                    "fix that includes best solutions from unfiltered results."
                )
            else:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in any available data."
                )
            return None

        plot_data = self.pareto_result.plot_data_collect[solution_id]
        adstock_data = plot_data["plot3data"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        # Handle different adstock types
        if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
            # Avoid copying if only sorting is done
            dt_geometric = adstock_data["dt_geometric"]

            # Sort data alphabetically by channel
            dt_geometric = dt_geometric.sort_values("channels", ascending=True)

            bars = ax.barh(
                y=range(len(dt_geometric)),
                width=dt_geometric["thetas"],
                height=0.5,
                color="coral",
            )

            for i, theta in enumerate(dt_geometric["thetas"]):
                ax.text(
                    theta + 0.01, i, f"{theta*100:.1f}%", va="center", fontweight="bold"
                )

            ax.set_yticks(range(len(dt_geometric)))
            ax.set_yticklabels(dt_geometric["channels"])

            # Format x-axis with 25% increments
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
            )
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.25, 0.25))  # Changed to 0.25 increments

            # Set title and labels
            interval_type = (
                self.mmm_data.mmmdata_spec.interval_type if self.mmm_data else "day"
            )
            ax.set_title(
                f"Geometric Adstock: Fixed Rate Over Time (Solution {solution_id})"
            )
            ax.set_xlabel(f"Thetas [by {interval_type}]")
            ax.set_ylabel(None)

        elif self.hyperparameter.adstock in [
            AdstockType.WEIBULL_CDF,
            AdstockType.WEIBULL_PDF,
        ]:
            # [Weibull code remains the same]
            weibull_data = adstock_data["weibullCollect"]
            wb_type = adstock_data["wb_type"]

            channels = sorted(
                weibull_data["channel"].unique()
            )  # Sort channels alphabetically
            rows = (len(channels) + 2) // 3

            if ax is None:
                fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
                axes = axes.flatten()
            else:
                gs = ax.get_gridspec()
                subfigs = ax.figure.subfigures(rows, 3)
                axes = [subfig.subplots() for subfig in subfigs]
                axes = [ax for sublist in axes for ax in sublist]

            for idx, channel in enumerate(channels):
                ax_sub = axes[idx]
                channel_data = weibull_data[weibull_data["channel"] == channel]

                ax_sub.plot(
                    channel_data["x"],
                    channel_data["decay_accumulated"],
                    color="steelblue",
                )

                ax_sub.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
                ax_sub.text(
                    max(channel_data["x"]),
                    0.5,
                    "Halflife",
                    color="gray",
                    va="bottom",
                    ha="right",
                )

                ax_sub.set_title(channel)
                ax_sub.grid(True, alpha=0.2)
                ax_sub.set_ylim(0, 1)

        # Customize grid
        if self.hyperparameter.adstock == AdstockType.GEOMETRIC:
            ax.grid(True, axis="x", alpha=0.2)
            ax.grid(False, axis="y")
        ax.set_axisbelow(True)

        ax.set_facecolor("white")

        logger.debug("Successfully generated adstock plot")

        if fig:
            plt.tight_layout()
            fig = plt.gcf()
            # Add metrics text if available
            if metrics:
                metrics_to_display = {
                    k: metrics.get(k, float("nan"))  # Use NaN for missing metrics
                    for k in [
                        "rsq_train",
                        "rsq_val",
                        "rsq_test",
                        "nrmse",
                        "nrmse_train",
                        "nrmse_val",
                        "nrmse_test",
                        "decomp.rssd",
                    ]
                }
                metrics_str_lines = [
                    f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
                    f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
                    f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
                ]
                # Place text below the title
                fig.text(
                    0.5,
                    0.95,
                    f"Metrics for Solution {solution_id}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    weight="bold",
                )
                fig.text(
                    0.5,
                    0.93,
                    metrics_str_lines[0],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.91,
                    metrics_str_lines[1],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                fig.text(
                    0.5,
                    0.89,
                    metrics_str_lines[2],
                    ha="center",
                    va="top",
                    fontsize=9,
                    color="grey",
                )
                # Adjust top margin to make space for the text
                plt.subplots_adjust(top=0.84)  # Adjust top margin

            plt.close(fig)
            return fig
        return None

    def create_prophet_decomposition_plot(self):
        """Create Prophet Decomposition Plot."""
        prophet_vars = (
            [ProphetVariableType(var) for var in self.holiday_data.prophet_vars]
            if self.holiday_data and self.holiday_data.prophet_vars
            else []
        )
        # Ensure factor_vars is always a list, never None
        factor_vars = (
            (self.mmm_data.mmmdata_spec.factor_vars or []) if self.mmm_data else []
        )

        if not (prophet_vars or factor_vars):
            return None

        # Avoid copying the dataframe if it's only used for reading/melting
        df = self.featurized_mmm_data.dt_mod
        prophet_vars_str = [variable.value for variable in prophet_vars]
        prophet_vars_str.sort(reverse=True)

        # Handle the case where self.mmm_data.mmmdata_spec.dep_var might be None
        dep_var = "dep_var"
        if hasattr(df, "dep_var"):
            dep_var = "dep_var"
        elif (
            self.mmm_data
            and hasattr(self.mmm_data.mmmdata_spec, "dep_var")
            and self.mmm_data.mmmdata_spec.dep_var is not None
        ):
            dep_var = self.mmm_data.mmmdata_spec.dep_var

        value_variables = [dep_var] + factor_vars + prophet_vars_str

        df_long = df.melt(
            id_vars=["ds"],
            value_vars=value_variables,
            var_name="variable",
            value_name="value",
        )
        df_long["ds"] = pd.to_datetime(df_long["ds"])
        plt.figure(figsize=(12, 3 * len(df_long["variable"].unique())))
        prophet_decomp_plot = plt.figure(
            figsize=(12, 3 * len(df_long["variable"].unique()))
        )
        gs = prophet_decomp_plot.add_gridspec(len(df_long["variable"].unique()), 1)
        for i, var in enumerate(df_long["variable"].unique()):
            ax = prophet_decomp_plot.add_subplot(gs[i, 0])
            var_data = df_long[df_long["variable"] == var]
            ax.plot(var_data["ds"], var_data["value"], color="steelblue")
            ax.set_title(var)
            ax.set_xlabel(None)
            ax.set_ylabel(None)
        plt.suptitle("Prophet decomposition")
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_hyperparameter_sampling_distribution(self):
        """Create Hyperparameter Sampling Distribution Plot."""
        unfiltered_pareto_results = self.unfiltered_pareto_result
        if unfiltered_pareto_results is None:
            return None
        result_hyp_param = unfiltered_pareto_results.result_hyp_param
        hp_names = list(self.hyperparameter.hyperparameters.keys())
        hp_names = [name.replace("lambda", "lambda_hp") for name in hp_names]
        matching_columns = [
            col
            for col in result_hyp_param.columns
            if any(re.search(pattern, col, re.IGNORECASE) for pattern in hp_names)
        ]
        matching_columns.sort()
        hyp_df = result_hyp_param[matching_columns]
        melted_df = hyp_df.melt(var_name="variable", value_name="value")
        melted_df["variable"] = melted_df["variable"].replace("lambda_hp", "lambda")

        def parse_variable(variable):
            parts = variable.split("_")
            return {"type": parts[-1], "channel": "_".join(parts[:-1])}

        parsed_vars = melted_df["variable"].apply(parse_variable).apply(pd.Series)
        melted_df[["type", "channel"]] = parsed_vars
        melted_df["type"] = pd.Categorical(
            melted_df["type"], categories=melted_df["type"].unique()
        )
        melted_df["channel"] = pd.Categorical(
            melted_df["channel"], categories=melted_df["channel"].unique()[::-1]
        )
        plt.figure(figsize=(12, 7))
        g = sns.FacetGrid(melted_df, col="type", sharex=False, height=6, aspect=1)

        def violin_plot(x, y, **kwargs):
            sns.violinplot(x=x, y=y, **kwargs, alpha=0.8, linewidth=0)

        g.map_dataframe(
            violin_plot, x="value", y="channel", hue="channel", palette="Set2"
        )
        g.set_titles("{col_name}")
        g.set_xlabels("Hyperparameter space")
        g.set_ylabels("")
        g.figure.suptitle("Hyperparameters Optimization Distributions", y=1.05)
        subtitle_text = (
            f"Sample distribution, iterations = "
            f"{self.model_outputs.iterations} x {len(self.model_outputs.trials)} trial"
        )
        g.figure.text(0.5, 0.98, subtitle_text, ha="center", fontsize=10)
        plt.subplots_adjust(top=0.9)
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_pareto_front_plot(self, is_calibrated):
        """Create Pareto Front Plot."""
        unfiltered_pareto_results = self.unfiltered_pareto_result
        # Only copy if modifications are needed (is_calibrated=True) to prevent side effects
        if is_calibrated:
            result_hyp_param = unfiltered_pareto_results.result_hyp_param.copy()
            result_hyp_param["iterations"] = np.where(
                result_hyp_param["robynPareto"].isna(),
                np.nan,
                result_hyp_param["iterations"],
            )
            result_hyp_param = result_hyp_param.sort_values(
                by="robynPareto", na_position="first"
            )
        else:
            # Use the original dataframe reference if no modifications are needed
            result_hyp_param = unfiltered_pareto_results.result_hyp_param

        pareto_fronts = self.pareto_result.pareto_fronts
        pareto_fronts_vec = list(range(1, pareto_fronts + 1))
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            result_hyp_param["nrmse"],
            result_hyp_param["decomp.rssd"],
            c=result_hyp_param["iterations"],
            cmap="Blues",
            alpha=0.7,
        )
        plt.colorbar(scatter, label="Iterations")
        if is_calibrated:
            scatter = plt.scatter(
                result_hyp_param["nrmse"],
                result_hyp_param["decomp.rssd"],
                c=result_hyp_param["iterations"],
                cmap="Blues",
                s=result_hyp_param["mape"] * 100,
                alpha=1 - result_hyp_param["mape"],
            )
        for pfs in range(1, max(pareto_fronts_vec) + 1):
            temp = result_hyp_param[result_hyp_param["robynPareto"] == pfs]
            if len(temp) > 1:
                temp = temp.sort_values("nrmse")
                plt.plot(temp["nrmse"], temp["decomp.rssd"], color="coral", linewidth=2)
        plt.title(
            "Multi-objective Evolutionary Performance"
            + (" with Calibration" if is_calibrated else "")
        )
        plt.xlabel("NRMSE")
        plt.ylabel("DECOMP.RSSD")
        plt.suptitle(
            f"2D Pareto fronts with {self.model_outputs.nevergrad_algo or 'Unknown'}, "
            f"for {len(self.model_outputs.trials)} trial{'' if pareto_fronts == 1 else 's'} "
            f"with {self.model_outputs.iterations or 1} iterations each"
        )
        plt.tight_layout()
        fig = plt.gcf()
        plt.close(fig)
        return fig

    def create_ridgeline_model_convergence(self):
        """Create Ridgeline Model Convergence Plots."""
        all_plots = {}
        x_decomp_agg = self.unfiltered_pareto_result.x_decomp_agg
        paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends
        dt_ridges = x_decomp_agg[
            x_decomp_agg["rn"].isin(paid_media_spends)
        ].copy()  # TODO get rid of this copy
        dt_ridges["iteration"] = (
            dt_ridges["iterNG"] - 1
        ) * self.model_outputs.cores + dt_ridges["iterPar"]
        dt_ridges = dt_ridges[["rn", "roi_total", "iteration", "trial"]]
        dt_ridges = dt_ridges.sort_values(["iteration", "rn"])
        iterations = self.model_outputs.iterations or 100
        qt_len = (
            1
            if iterations <= 100
            else (20 if iterations > 2000 else int(np.ceil(iterations / 100)))
        )
        set_qt = np.floor(np.linspace(1, iterations, qt_len + 1)).astype(int)
        set_bin = set_qt[1:]
        dt_ridges["iter_bin"] = pd.cut(
            dt_ridges["iteration"], bins=set_qt, labels=set_bin
        )
        dt_ridges = dt_ridges.dropna(subset=["iter_bin"])
        dt_ridges["iter_bin"] = pd.Categorical(
            dt_ridges["iter_bin"],
            categories=sorted(set_bin, reverse=True),
            ordered=True,
        )
        dt_ridges["trial"] = dt_ridges["trial"].astype("category")
        plot_vars = dt_ridges["rn"].unique()
        plot_n = int(np.ceil(len(plot_vars) / 6))
        metric = (
            "ROAS"
            if self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
            else "CPA"
        )
        for pl in range(1, plot_n + 1):
            start_idx = (pl - 1) * 6
            loop_vars = plot_vars[start_idx : start_idx + 6]
            dt_ridges_loop = dt_ridges[dt_ridges["rn"].isin(loop_vars)]
            fig, axes = plt.subplots(
                nrows=len(loop_vars), figsize=(12, 3 * len(loop_vars)), sharex=False
            )
            if len(loop_vars) == 1:
                axes = [axes]
            for idx, var in enumerate(loop_vars):
                var_data = dt_ridges_loop[dt_ridges_loop["rn"] == var]
                offset = 0
                for iter_bin in sorted(var_data["iter_bin"].unique(), reverse=True):
                    bin_data = var_data[var_data["iter_bin"] == iter_bin]["roi_total"]
                    sns.kdeplot(
                        bin_data,
                        ax=axes[idx],
                        fill=True,
                        alpha=0.6,
                        color=plt.cm.GnBu(offset / len(var_data["iter_bin"].unique())),
                        label=f"Bin {iter_bin}",
                        warn_singular=False,
                    )
                    offset += 1
                axes[idx].set_title(f"{var} {metric}")
                axes[idx].set_ylabel("")
                axes[idx].spines["right"].set_visible(False)
                axes[idx].spines["top"].set_visible(False)
            plt.suptitle(f"{metric} Distribution over Iteration Buckets", fontsize=16)
            plt.tight_layout()
            fig = plt.gcf()
            plt.close(fig)
            all_plots[f"{metric}_convergence_{pl}"] = fig
        return all_plots

    def plot_all(
        self,
        display_plots: bool = True,
        export_location: Union[str, Path] = None,
        display_criteria: str = "best_rsq_train",
    ) -> None:
        """
        Generates and manages plots for Pareto results based on specified criteria,
        optimized for memory efficiency.

        Plots are generated iteratively. If exporting, they are saved immediately.
        If displaying, only the figures matching `display_criteria` are kept in memory
        temporarily for display. All figures are closed after processing to release memory.

        Args:
            display_plots: If True, displays the plots selected by `display_criteria`.
            export_location: Path to export all generated plots. If provided, plots
                             are saved iteratively.
            display_criteria: Key specifying which set of plots to display.
                              Valid keys: 'best_rsq_train', 'best_rsq_test',
                              'best_nrmse_train', 'best_nrmse_test', or keys of
                              non-solution specific plots like 'prophet_decomp'.
                              Defaults to 'best_rsq_train'.
        """
        figures_to_display: Dict[str, plt.Figure] = {}
        target_solutions: Dict[str, Optional[str]] = (
            {}
        )  # To store solution IDs based on criteria
        export_path: Optional[Path] = None

        if export_location:
            export_path = Path(export_location)
            export_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Exporting plots enabled. Target directory: {export_path}")

        # Helper function to safely get the best solution ID (remains the same)
        def get_best_sol_id(df, metric, ascending) -> Optional[str]:
            if df is None or df.empty or metric not in df.columns:
                logger.warning(
                    f"Metric '{metric}' not found or DataFrame is empty for identifying best solution."
                )
                return None
            try:
                df_filtered = df.dropna(subset=[metric])
                if df_filtered.empty:
                    logger.warning(
                        f"No valid data for metric '{metric}' after dropping NaNs."
                    )
                    return None
                sorted_df = df_filtered.sort_values(metric, ascending=ascending)
                sol_id_col = next(
                    (col for col in ["sol_id", "solID"] if col in sorted_df.columns),
                    None,
                )
                index_is_sol_id = sorted_df.index.name in ["sol_id", "solID"]

                if sol_id_col:
                    best_id = sorted_df.iloc[0][sol_id_col]
                elif index_is_sol_id:
                    best_id = sorted_df.index[0]
                else:
                    logger.warning(
                        f"Cannot determine 'sol_id' or 'solID' in DataFrame for metric '{metric}'. Using first index."
                    )
                    best_id = sorted_df.index[0]
                return str(best_id)
            except IndexError:
                logger.warning(
                    f"IndexError while getting best solution for {metric}. DataFrame might be empty after filtering."
                )
                return None
            except Exception as e:
                logger.warning(f"Error getting best solution for {metric}: {e}")
                return None

        # --- Identify Target Solutions (remains the same) ---
        cleaned_results_df = self.pareto_result.result_hyp_param
        all_results_df = (
            self.unfiltered_pareto_result.result_hyp_param
            if self.unfiltered_pareto_result
            else None
        )

        criteria_metrics = {
            "best_rsq_train": ("rsq_train", False),
            "best_rsq_test": ("rsq_test", False),
            "best_nrmse_train": ("nrmse_train", True),
            "best_nrmse_test": ("nrmse_test", True),
        }

        for key, (metric, ascending) in criteria_metrics.items():
            # Try finding in cleaned results first, fallback to all results if needed/available
            best_id = get_best_sol_id(cleaned_results_df, metric, ascending)
            # Optional: Fallback to all_results_df if not found in cleaned? Depends on desired behavior.
            # if best_id is None and all_results_df is not None:
            #     best_id = get_best_sol_id(all_results_df, metric, ascending)
            target_solutions[key] = best_id
            logger.info(f"Identified solution for {key}: {target_solutions[key]}")

        solution_ids_to_plot = sorted(
            list(set(filter(None, target_solutions.values())))
        )

        # Determine the specific solution ID to display plots for
        display_sol_id = (
            target_solutions.get(display_criteria)
            if display_criteria in target_solutions
            else None
        )
        logger.info(
            f"Display criteria '{display_criteria}' targets solution ID: {display_sol_id}"
        )

        # --- Fetch Metrics Data (remains similar, ensure robust indexing) ---
        metric_cols = [
            "rsq_train",
            "rsq_val",
            "rsq_test",
            "nrmse",
            "nrmse_train",
            "nrmse_val",
            "nrmse_test",
            "decomp.rssd",
        ]
        metrics_data_all_indexed = None  # Use a version indexed by sol_id
        if all_results_df is not None and not all_results_df.empty:
            sol_id_col = next(
                (col for col in ["sol_id", "solID"] if col in all_results_df.columns),
                None,
            )
            index_is_sol_id = all_results_df.index.name in ["sol_id", "solID"]

            if index_is_sol_id:
                metrics_data_all_indexed = all_results_df
            elif sol_id_col:
                try:
                    # Avoid modifying original df, ensure index contains unique IDs
                    if all_results_df[sol_id_col].is_unique:
                        metrics_data_all_indexed = all_results_df.set_index(
                            sol_id_col, drop=False
                        )
                    else:
                        logger.warning(
                            f"Solution ID column '{sol_id_col}' is not unique. Cannot reliably index by it. Metrics might be inaccurate."
                        )
                        # Fallback or handle duplicates if necessary
                        metrics_data_all_indexed = (
                            all_results_df  # Keep original, lookup will be slower
                        )
                except Exception as e:
                    logger.warning(
                        f"Error setting '{sol_id_col}' as index: {e}. Proceeding with original structure."
                    )
                    metrics_data_all_indexed = all_results_df  # Fallback
            else:
                logger.warning(
                    "Could not find 'sol_id' or 'solID' as index or column in unfiltered results. Metrics might not be displayed."
                )
                metrics_data_all_indexed = all_results_df  # Fallback
        else:
            logger.warning(
                "Unfiltered results DataFrame is missing or empty. Cannot fetch metrics."
            )

        # --- Helper to process (save/display) a generated figure ---
        def process_figure(
            fig: Optional[plt.Figure],
            plot_key: str,
            is_solution_plot: bool,
            sol_id: Optional[str] = None,
        ):
            if not fig:
                logger.warning(
                    f"Plot function returned None for key '{plot_key}'"
                    + (f" (Solution: {sol_id})" if sol_id else "")
                )
                return

            should_display = False
            if display_plots:
                if is_solution_plot:
                    # Display if it's the target solution and the criteria matches the plot's origin criteria
                    criteria_keys_for_sol = [
                        k for k, v in target_solutions.items() if v == sol_id
                    ]
                    if (
                        sol_id == display_sol_id
                        and display_criteria in criteria_keys_for_sol
                    ):
                        should_display = True
                elif (
                    plot_key == display_criteria
                ):  # Display non-solution plot if its key matches criteria
                    should_display = True

            saved = False
            if export_path:
                try:
                    filename = export_path / f"{plot_key}.png"
                    fig.savefig(filename, bbox_inches="tight")
                    logger.debug(f"Exported plot: {filename}")
                    saved = True
                except Exception as e:
                    logger.error(
                        f"Error exporting plot '{plot_key}': {e}", exc_info=True
                    )

            if should_display:
                figures_to_display[plot_key] = fig  # Keep figure open for display
            else:
                plt.close(fig)  # Close figure if not needed for display

        # --- Generate Non-Solution Specific Plots ---
        if not self.model_outputs.hyper_fixed:
            non_solution_plot_funcs = {
                "prophet_decomp": self.create_prophet_decomposition_plot,
                "hyperparameters_sampling": self.create_hyperparameter_sampling_distribution,
                "pareto_front": lambda: self.create_pareto_front_plot(
                    is_calibrated=False
                ),  # Use lambda for args
                # Ridgeline returns a dict, handle separately
            }
            for key, func in non_solution_plot_funcs.items():
                try:
                    fig = func()
                    process_figure(fig, key, is_solution_plot=False)
                except Exception as e:
                    logger.error(
                        f"Error generating non-solution plot '{key}': {e}",
                        exc_info=True,
                    )

            # Handle Ridgeline separately as it returns multiple plots
            try:
                ridgeline_plots = self.create_ridgeline_model_convergence()
                if ridgeline_plots:
                    for key, fig in ridgeline_plots.items():
                        process_figure(fig, key, is_solution_plot=False)
            except Exception as e:
                logger.error(
                    f"Error generating ridgeline convergence plots: {e}", exc_info=True
                )

        # --- Generate Plots for Unique Target Solutions ---
        if not solution_ids_to_plot:
            logger.warning(
                "Could not find any valid solution IDs based on the criteria. No solution-specific plots will be generated."
            )
        else:
            logger.info(
                f"Generating plots for unique solution IDs: {solution_ids_to_plot}"
            )
            plot_funcs_solution = {
                "waterfall": self.generate_waterfall,
                "fitted_vs_actual": self.generate_fitted_vs_actual,
                "diagnostic_plot": self.generate_diagnostic_plot,
                "immediate_vs_carryover": self.generate_immediate_vs_carryover,
                "adstock_rate": self.generate_adstock_rate,
            }

            for solution_id in solution_ids_to_plot:
                solution_metrics = {}
                if metrics_data_all_indexed is not None:
                    # Check which metric columns actually exist in the dataframe
                    available_metrics = [
                        col
                        for col in metric_cols
                        if col in metrics_data_all_indexed.columns
                    ]
                    try:
                        # Use .loc for index-based lookup if indexed, otherwise filter
                        if (
                            metrics_data_all_indexed.index.name in ["sol_id", "solID"]
                            and solution_id in metrics_data_all_indexed.index
                        ):
                            solution_metrics = metrics_data_all_indexed.loc[
                                solution_id, available_metrics
                            ].to_dict()
                        else:  # Fallback to filtering if not indexed or ID not in index
                            sol_id_col_lookup = next(
                                (
                                    col
                                    for col in ["sol_id", "solID"]
                                    if col in metrics_data_all_indexed.columns
                                ),
                                None,
                            )
                            if sol_id_col_lookup:
                                row = metrics_data_all_indexed[
                                    metrics_data_all_indexed[sol_id_col_lookup]
                                    == solution_id
                                ]
                                if not row.empty:
                                    solution_metrics = row.iloc[0][
                                        available_metrics
                                    ].to_dict()
                                else:
                                    logger.warning(
                                        f"Solution ID {solution_id} not found via filtering."
                                    )
                            else:
                                logger.warning(
                                    f"Cannot find solution ID {solution_id} as index or column."
                                )

                    except KeyError:
                        logger.warning(
                            f"KeyError fetching metrics for solution {solution_id}. Skipping metrics display for this solution."
                        )
                    except Exception as e:
                        logger.warning(
                            f"Error fetching metrics for solution {solution_id}: {e}. Skipping metrics display."
                        )
                else:
                    logger.warning(
                        f"Metrics data not available for solution ID {solution_id}."
                    )

                # Find the criteria keys associated with this solution_id
                criteria_keys = sorted(
                    [k for k, v in target_solutions.items() if v == solution_id]
                )  # Sort for consistent naming
                criteria_str = "_".join(criteria_keys)

                for plot_name, plot_func in plot_funcs_solution.items():
                    full_name = f"({criteria_str})__{plot_name}_{solution_id}"
                    try:
                        fig = plot_func(
                            solution_id=solution_id, metrics=solution_metrics
                        )
                        process_figure(
                            fig, full_name, is_solution_plot=True, sol_id=solution_id
                        )
                    except Exception as e:
                        logger.error(
                            f"Error generating plot '{full_name}' for solution {solution_id}: {e}",
                            exc_info=True,
                        )

        # --- Display Collected Plots ---
        if display_plots:
            if figures_to_display:
                logger.info(
                    f"Displaying {len(figures_to_display)} plots for criteria '{display_criteria}'"
                    + (f" (Solution ID: {display_sol_id})" if display_sol_id else "")
                )
                try:
                    # Assuming self.display_plots() handles showing the figures
                    self.display_plots(figures_to_display)
                except Exception as e:
                    logger.error(f"Error during plot display: {e}", exc_info=True)
                finally:
                    # Ensure all displayed figures are closed afterwards
                    for key, fig in figures_to_display.items():
                        try:
                            plt.close(fig)
                        except Exception as e:
                            logger.warning(f"Error closing displayed plot '{key}': {e}")
                    figures_to_display.clear()  # Clear the dictionary
            else:
                logger.warning(
                    f"No plots were collected to display for criteria '{display_criteria}'."
                )

        logger.info("Finished plot generation and processing.")
