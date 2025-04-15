# pyre-strict
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robyn.data.entities.enums import DependentVarType
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class TransformationVisualizer(BaseVisualizer):
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        logger.debug(
            "Initializing TransformationVisualizer with pareto_result=%s, mmm_data=%s",
            pareto_result,
            mmm_data,
        )
        super().__init__()
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def create_adstock_plots(self) -> None:
        """
        Generate adstock visualization plots and store them as instance variables.
        """
        logger.info("Starting creation of adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Adstock plots creation completed successfully")
        except Exception as e:
            logger.error("Failed to create adstock plots: %s", str(e))
            raise

    def create_saturation_plots(self) -> None:
        """
        Generate saturation visualization plots and store them as instance variables.
        """
        logger.info("Starting creation of saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Saturation plots creation completed successfully")
        except Exception as e:
            logger.error("Failed to create saturation plots: %s", str(e))
            raise

    def get_adstock_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the adstock plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for adstock plots
        """
        logger.debug("Retrieving adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully retrieved adstock plots")
            return None
        except Exception as e:
            logger.error("Failed to retrieve adstock plots: %s", str(e))
            raise

    def get_saturation_plots(self) -> Optional[Tuple[plt.Figure, plt.Figure]]:
        """
        Retrieve the saturation plots.

        Returns:
            Optional[Tuple[plt.Figure, plt.Figure]]: Tuple of matplotlib figures for saturation plots
        """
        logger.debug("Retrieving saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully retrieved saturation plots")
            return None
        except Exception as e:
            logger.error("Failed to retrieve saturation plots: %s", str(e))
            raise

    def display_adstock_plots(self) -> None:
        """
        Display the adstock plots.
        """
        logger.info("Displaying adstock plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully displayed adstock plots")
        except Exception as e:
            logger.error("Failed to display adstock plots: %s", str(e))
            raise

    def display_saturation_plots(self) -> None:
        """
        Display the saturation plots.
        """
        logger.info("Displaying saturation plots")
        try:
            # Implementation placeholder
            logger.debug("Successfully displayed saturation plots")
        except Exception as e:
            logger.error("Failed to display saturation plots: %s", str(e))
            raise

    def save_adstock_plots(self, filenames: List[str]) -> None:
        """
        Save the adstock plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        logger.info("Saving adstock plots to files: %s", filenames)
        try:
            # Implementation placeholder
            logger.debug("Successfully saved adstock plots")
        except Exception as e:
            logger.error("Failed to save adstock plots: %s", str(e))
            raise

    def save_saturation_plots(self, filenames: List[str]) -> None:
        """
        Save the saturation plots to files.

        Args:
            filenames (List[str]): List of filenames to save the plots
        """
        logger.info("Saving saturation plots to files: %s", filenames)
        try:
            # Implementation placeholder
            logger.debug("Successfully saved saturation plots")
        except Exception as e:
            logger.error("Failed to save saturation plots: %s", str(e))
            raise

    def generate_spend_effect_comparison(
        self, solution_id: str, ax: Optional[plt.Axes] = None
    ) -> Optional[plt.Figure]:
        """Generate comparison plot of spend share vs effect share."""

        logger.debug("Starting generation of spend effect comparison plot")
        try:
            # Get plot data safely
            logger.debug("Extracting plot data from pareto result")
            plot_data = self.pareto_result.plot_data_collect[solution_id]

            # Safely get bar and line data
            try:
                bar_data = plot_data["plot1data"]["plotMediaShareLoopBar"].copy()
                line_data = plot_data["plot1data"]["plotMediaShareLoopLine"].copy()
                y_sec_scale = plot_data["plot1data"]["ySecScale"]

                logger.debug(
                    "Processing plot data - bar_data shape: %s, line_data shape: %s",
                    bar_data.shape,
                    line_data.shape,
                )

                # Convert y_sec_scale to float safely
                if isinstance(y_sec_scale, pd.DataFrame):
                    y_sec_scale = float(
                        y_sec_scale.iat[0, 0]
                        if len(y_sec_scale.columns) > 0
                        else y_sec_scale.iloc[0]
                    )
                elif isinstance(y_sec_scale, pd.Series):
                    y_sec_scale = float(y_sec_scale.iloc[0])
                else:
                    y_sec_scale = float(y_sec_scale)

                logger.debug("Y-scale factor: %f", y_sec_scale)
            except (KeyError, AttributeError, IndexError) as e:
                logger.error(
                    "Error accessing plot data for solution %s: %s", solution_id, str(e)
                )
                return None

            # Transform variable names safely
            bar_data["variable"] = (
                bar_data["variable"].str.replace("_", " ").str.title()
            )

            # Create figure if no axes provided
            if ax is None:
                logger.debug("Creating new figure and axes")
                fig, ax = plt.subplots(figsize=(16, 10))
                plt.subplots_adjust(top=0.80, left=0.15, bottom=0.1, right=0.95)
            else:
                logger.debug("Using provided axes for plotting")
                fig = None

            # Set background color
            ax.set_facecolor("white")

            # Set up colors
            type_colour = "#03396C"  # Dark blue for line
            bar_colors = ["#A4C2F4", "#FFB7B2"]  # Light blue and light coral for bars
            bar_colors = bar_colors[::-1]  # Reverse colors

            # Set up dimensions
            channels = sorted(
                line_data["rn"].unique()
            )  # Use line_data for consistent ordering
            y_pos = np.arange(len(channels))

            logger.debug("Processing %d channels for visualization", len(channels))

            # Plot bars for each variable type
            bar_width = 0.35
            for i, (var, color) in enumerate(
                zip(reversed(bar_data["variable"].unique()), bar_colors)
            ):
                var_data = bar_data[bar_data["variable"] == var]
                # Ensure alignment with channels - safely get values
                values = []
                for ch in channels:
                    ch_data = var_data[var_data["rn"] == ch]
                    if not ch_data.empty:
                        values.append(ch_data["value"].iloc[0])
                    else:
                        values.append(0)

                logger.debug(
                    "Plotting bars for variable '%s' with %d values", var, len(values)
                )
                bars = ax.barh(
                    y=[y + (i - 0.5) * bar_width for y in y_pos],
                    width=values,
                    height=bar_width,
                    label=var,
                    color=color,
                    alpha=0.5,
                )

                # Add percentage labels to right of y-axis
                for idx, value in enumerate(values):
                    y_position = y_pos[idx] + (i - 0.5) * bar_width
                    percentage = f"{value * 100:.1f}%"

                    ax.text(
                        s=percentage,
                        x=0.02,
                        y=y_position,
                        ha="left",
                        va="center",
                        fontweight="bold",
                        fontsize=9,
                        transform=ax.get_yaxis_transform(),
                    )

            # Safely get line values
            line_values = []
            for ch in channels:
                ch_data = line_data[line_data["rn"] == ch]
                if not ch_data.empty:
                    line_values.append(ch_data["value"].iloc[0])
                else:
                    line_values.append(0)

            line_values = np.array(line_values)
            line_x = line_values / y_sec_scale

            logger.debug("Plotting line with %d points", len(line_x))
            # Plot line without label
            ax.plot(
                line_x, y_pos, color=type_colour, marker="o", markersize=8, zorder=3
            )

            # Add line value labels
            for i, value in enumerate(line_values):
                ax.text(
                    line_x[i] + 0.02,  # Added offset to move labels right of dots
                    y_pos[i],
                    f"{value:.2f}",
                    color=type_colour,
                    fontweight="bold",
                    ha="left",  # Left align since we're positioning to the right of dots
                    va="center",
                    zorder=4,
                )

            # Set channel labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(channels)
            ax.tick_params(axis="y", pad=5)

            # Format x-axis as percentage and show labels up to 70%
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%")
            )
            ax.set_xlim(0, 0.7)  # Set x-axis limit to 70%
            xticks = np.arange(
                0, 0.8, 0.1
            )  # Create ticks from 0 to 70% in 10% increments
            ax.set_xticks(xticks)
            ax.tick_params(axis="x", labelbottom=True)

            # Add grid
            ax.grid(True, axis="x", alpha=0.2, linestyle="-")
            ax.set_axisbelow(True)

            # Remove unnecessary spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Set title at top position
            metric_type = (
                "ROAS"
                if (
                    self.mmm_data
                    and hasattr(self.mmm_data.mmmdata_spec, "dep_var_type")
                    and self.mmm_data.mmmdata_spec.dep_var_type
                    == DependentVarType.REVENUE
                )
                else "CPA"
            )

            logger.debug("Setting plot title with metric type: %s", metric_type)

            # Set title at top
            ax.set_title(
                f"Share of Total Spend, Effect & {metric_type} in Modeling Window*",
                pad=20,
                y=1.45,
            )

            # Create legend with single ROAS entry
            bars_legend = ax.get_legend_handles_labels()
            line_legend = [
                plt.Line2D(
                    [0],
                    [0],
                    color=type_colour,
                    marker="o",
                    linestyle="-",
                    markersize=8,
                    label="ROAS",
                )
            ]

            # Combine legend elements in reverse order
            handles = line_legend + list(reversed(bars_legend[0]))
            labels = ["ROAS"] + list(reversed(bars_legend[1]))

            # Create legend below title
            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(0, 1.05, 0.4, 0.1),
                loc="lower left",
                mode="expand",
                ncol=3,
                frameon=False,
                borderaxespad=0,
            )

            # Add axis labels
            ax.set_xlabel("Total Share by Channel")
            ax.set_ylabel(None)

            logger.debug("Successfully generated spend effect comparison plot")
            if fig:
                plt.tight_layout()
                plt.subplots_adjust(top=0.80, left=0.15, bottom=0.1, right=0.95)
                return fig
            return None

        except (KeyError, AttributeError, IndexError, ValueError) as error:
            logger.error(
                "Error generating spend effect plot for solution %s: %s",
                solution_id,
                str(error),
            )
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    "Error generating spend effect plot",
                    ha="center",
                    va="center",
                )
            return None

    def _get_last_calendar_year_quarters(self) -> List[List[str]]:
        """
        Get date ranges for the four quarters of the last complete calendar year.

        Returns:
            List of four date ranges, each containing [start_date, end_date] as strings.
        """
        # Get the available dates from the data
        ts_data = None

        # Try to get dates from mmm_data
        if (
            self.mmm_data
            and hasattr(self.mmm_data, "dt")
            and "ds" in self.mmm_data.dt.columns
        ):
            # Get dates directly from mmm_data
            ts_data = pd.to_datetime(self.mmm_data.dt["ds"].unique())

        if ts_data is None or len(ts_data) == 0:
            logger.warning(
                "Could not find date information to determine last calendar year quarters."
            )
            # Return default dates for previous year's quarters (fallback)
            year = pd.Timestamp.now().year - 1
            return [
                [f"{year}-01-01", f"{year}-03-31"],
                [f"{year}-04-01", f"{year}-06-30"],
                [f"{year}-07-01", f"{year}-09-30"],
                [f"{year}-10-01", f"{year}-12-31"],
            ]

        # Find the most recent complete calendar year in the data
        max_date = pd.Timestamp(max(ts_data))
        min_date = pd.Timestamp(min(ts_data))

        # Find the most recent year that has all quarters (at least partially) covered
        last_year = max_date.year
        if (
            max_date.month < 12
        ):  # If current year doesn't have Q4 data, use previous year
            last_year -= 1

        # Ensure the chosen year has data
        if pd.Timestamp(f"{last_year}-01-01") < min_date:
            # If earliest data is after the start of the chosen year, adjust
            last_year = min_date.year

        # Define the quarters
        q1 = [f"{last_year}-01-01", f"{last_year}-03-31"]
        q2 = [f"{last_year}-04-01", f"{last_year}-06-30"]
        q3 = [f"{last_year}-07-01", f"{last_year}-09-30"]
        q4 = [f"{last_year}-10-01", f"{last_year}-12-31"]

        return [q1, q2, q3, q4]

    def _add_metrics_to_plot(
        self, fig: plt.Figure, metrics: Dict[str, float], solution_id: str
    ) -> None:
        """
        Helper method to add metrics text to a plot in the top right corner.

        Args:
            fig: matplotlib Figure object
            metrics: Dictionary of metric values
            solution_id: Solution ID string
        """
        if not metrics:
            return

        # Get metrics to display, using NaN for missing metrics
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
                "mae",
            ]
        }

        # Format metrics as text lines
        metrics_str_lines = [
            f"Metrics for Solution {solution_id}",
            f"Train R²: {metrics_to_display['rsq_train']:.3f}, Val R²: {metrics_to_display['rsq_val']:.3f}, Test R²: {metrics_to_display['rsq_test']:.3f}",
            f"NRMSE: {metrics_to_display['nrmse']:.3f} (Train: {metrics_to_display['nrmse_train']:.3f}, Val: {metrics_to_display['nrmse_val']:.3f}, Test: {metrics_to_display['nrmse_test']:.3f})",
            f"MAE: {metrics_to_display['mae']:.3f}",
            f"Decomp RSSD: {metrics_to_display['decomp.rssd']:.3f}",
        ]

        # Set position for metrics text (top right corner)
        x_pos = 0.98  # Right side of figure
        y_start = 0.98  # Top of figure
        line_spacing = 0.025  # Space between lines

        # Create a common bbox style for all text elements
        metrics_box = dict(
            facecolor="white",
            alpha=0.8,
            edgecolor="lightgray",
            boxstyle="round,pad=0.3",
        )

        # Add each metrics line with background box
        for i, line in enumerate(metrics_str_lines):
            # First line (title) is bold
            weight = "bold" if i == 0 else "normal"
            fontsize = 10 if i == 0 else 9
            color = "black" if i == 0 else "grey"

            fig.text(
                x_pos,
                y_start - (i * line_spacing),
                line,
                ha="right",  # Right-aligned text
                va="top",
                fontsize=fontsize,
                weight=weight,
                color=color,
                bbox=metrics_box,  # Add box to all lines for better visibility
            )

    def _generate_spend_effect_comparison_for_date_range(
        self,
        solution_id: str,
        date_range: List[str],
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """Generate comparison plot of spend share vs effect share for a specific date range.

        Args:
            solution_id: ID of the solution to visualize
            date_range: List of two elements [start_date, end_date] for filtering data
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            metrics: Optional dictionary containing model performance metrics

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object
        """
        logger.debug(
            "Starting generation of date-filtered spend effect comparison plot"
        )

        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(
            date_range[1]
        )

        # Extract plot data from pareto result
        try:
            plot_data = self.pareto_result.plot_data_collect[solution_id]
            bar_data = plot_data["plot1data"]["plotMediaShareLoopBar"].copy()
            line_data = plot_data["plot1data"]["plotMediaShareLoopLine"].copy()
            y_sec_scale = plot_data["plot1data"]["ySecScale"]

            # Convert y_sec_scale to float safely
            if isinstance(y_sec_scale, pd.DataFrame):
                y_sec_scale = float(
                    y_sec_scale.iat[0, 0]
                    if len(y_sec_scale.columns) > 0
                    else y_sec_scale.iloc[0]
                )
            elif isinstance(y_sec_scale, pd.Series):
                y_sec_scale = float(y_sec_scale.iloc[0])
            else:
                y_sec_scale = float(y_sec_scale)
        except (KeyError, AttributeError) as e:
            logger.error(
                f"Error accessing plot data for solution {solution_id}: {str(e)}"
            )
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    f"Error accessing plot data: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return None

        # Get model coefficients and model data to filter by date
        try:
            # Assuming x_decomp_vec_collect contains time series data with 'ds' column
            x_decomp_vec = self.pareto_result.x_decomp_vec_collect[
                self.pareto_result.x_decomp_vec_collect["sol_id"] == solution_id
            ]

            # Convert 'ds' to datetime and filter by date range
            x_decomp_vec.loc[:, "ds"] = pd.to_datetime(x_decomp_vec["ds"])
            date_filtered = x_decomp_vec[
                (x_decomp_vec["ds"] >= start_date) & (x_decomp_vec["ds"] <= end_date)
            ]

            if date_filtered.empty:
                logger.warning(
                    f"No data found for date range {start_date} to {end_date}"
                )
                if ax:
                    ax.text(
                        0.5,
                        0.5,
                        f"No data found for date range {date_range[0]} to {date_range[1]}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None
        except Exception as e:
            logger.error(f"Error filtering data by date range: {str(e)}")
            if ax:
                ax.text(
                    0.5,
                    0.5,
                    f"Error filtering data: {str(e)}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return None

        # Transform variable names
        bar_data["variable"] = bar_data["variable"].str.replace("_", " ").str.title()

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, 10))
        else:
            fig = None

        # Set background color
        ax.set_facecolor("white")

        # Set up colors
        type_colour = "#03396C"  # Dark blue for line
        bar_colors = ["#A4C2F4", "#FFB7B2"]  # Light blue and light coral for bars
        bar_colors = bar_colors[::-1]  # Reverse colors

        # Set up dimensions
        channels = sorted(line_data["rn"].unique())
        y_pos = np.arange(len(channels))

        # Plot bars for each variable type
        bar_width = 0.35
        for i, (var, color) in enumerate(
            zip(reversed(bar_data["variable"].unique()), bar_colors)
        ):
            var_data = bar_data[bar_data["variable"] == var]
            values = []
            for ch in channels:
                ch_data = var_data[var_data["rn"] == ch]
                if not ch_data.empty:
                    values.append(ch_data["value"].iloc[0])
                else:
                    values.append(0)

            bars = ax.barh(
                y=[y + (i - 0.5) * bar_width for y in y_pos],
                width=values,
                height=bar_width,
                label=var,
                color=color,
                alpha=0.5,
            )

            # Add percentage labels
            for idx, value in enumerate(values):
                y_position = y_pos[idx] + (i - 0.5) * bar_width
                percentage = f"{value * 100:.1f}%"

                ax.text(
                    s=percentage,
                    x=0.02,
                    y=y_position,
                    ha="left",
                    va="center",
                    fontweight="bold",
                    fontsize=9,
                    transform=ax.get_yaxis_transform(),
                )

        # Get line values
        line_values = []
        for ch in channels:
            ch_data = line_data[line_data["rn"] == ch]
            if not ch_data.empty:
                line_values.append(ch_data["value"].iloc[0])
            else:
                line_values.append(0)

        line_values = np.array(line_values)
        line_x = line_values / y_sec_scale

        # Plot line
        ax.plot(line_x, y_pos, color=type_colour, marker="o", markersize=8, zorder=3)

        # Add line value labels
        for i, value in enumerate(line_values):
            ax.text(
                line_x[i] + 0.02,
                y_pos[i],
                f"{value:.2f}",
                color=type_colour,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=4,
            )

        # Set channel labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(channels)
        ax.tick_params(axis="y", pad=5)

        # Format x-axis as percentage
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x*100:.0f}%"))
        ax.set_xlim(0, 0.7)
        xticks = np.arange(0, 0.8, 0.1)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelbottom=True)

        # Add grid
        ax.grid(True, axis="x", alpha=0.2, linestyle="-")
        ax.set_axisbelow(True)

        # Remove unnecessary spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Set title and labels
        metric_type = (
            "ROAS"
            if (
                self.mmm_data
                and hasattr(self.mmm_data.mmmdata_spec, "dep_var_type")
                and self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
            )
            else "CPA"
        )

        # Add date range to title
        date_range_str = f" ({date_range[0]} to {date_range[1]})"
        ax.set_title(
            f"Share of Total Spend, Effect & {metric_type}{date_range_str}",
            pad=20,
            y=1.05,
        )

        # Create legend
        bars_legend = ax.get_legend_handles_labels()
        line_legend = [
            plt.Line2D(
                [0],
                [0],
                color=type_colour,
                marker="o",
                linestyle="-",
                markersize=8,
                label=metric_type,
            )
        ]

        # Combine legend elements
        handles = line_legend + list(reversed(bars_legend[0]))
        labels = [metric_type] + list(reversed(bars_legend[1]))

        # Add legend
        ax.legend(
            handles=handles,
            labels=labels,
            bbox_to_anchor=(0, 1.05, 0.4, 0.1),
            loc="lower left",
            mode="expand",
            ncol=3,
            frameon=False,
            borderaxespad=0,
        )

        # Add axis labels
        ax.set_xlabel("Total Share by Channel")
        ax.set_ylabel(None)

        logger.debug(
            "Successfully generated date-filtered spend effect comparison plot"
        )
        return fig

    def generate_quarterly_spend_effect_comparison(
        self,
        solution_id: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[plt.Figure]:
        """
        Generate spend effect comparison charts for each quarter of the last calendar year in a 2x2 grid.

        Args:
            solution_id: ID of the solution to visualize
            metrics: Optional dictionary containing model performance metrics

        Returns:
            Optional[plt.Figure]: Generated matplotlib Figure object with 4 subplots
        """
        logger.debug("Starting generation of quarterly spend effect comparison plots")

        # Check if solution_id exists in the data
        if solution_id not in self.pareto_result.plot_data_collect:
            logger.warning(
                f"Invalid solution ID: {solution_id}. Solution not found in available data."
            )
            return None

        # Get quarterly date ranges
        quarters = self._get_last_calendar_year_quarters()
        quarter_names = ["Q1", "Q2", "Q3", "Q4"]
        year = quarters[0][0].split("-")[
            0
        ]  # Extract year from first quarter's start date

        # Create a figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Generate spend effect comparison for each quarter
        for i, (quarter_range, quarter_name) in enumerate(zip(quarters, quarter_names)):
            try:
                # Call method for date-filtered spend effect comparison
                self._generate_spend_effect_comparison_for_date_range(
                    solution_id=solution_id,
                    date_range=quarter_range,
                    ax=axes[i],
                    metrics=None,  # Don't add metrics to individual subplots
                )

                # Adjust subplot title to be cleaner
                quarter_title = f"{year} {quarter_name}"
                axes[i].set_title(quarter_title, pad=20, y=1.05)

            except Exception as e:
                logger.warning(
                    f"Error generating quarterly spend effect comparison for {quarter_name}: {e}"
                )
                axes[i].text(
                    0.5,
                    0.5,
                    f"Error generating {quarter_name} spend effect comparison",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                # Add error details
                axes[i].text(
                    0.5,
                    0.4,
                    str(e),
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                    fontsize=9,
                    color="gray",
                    wrap=True,
                )

        # Add overall title
        fig.suptitle(
            f"Quarterly Spend Effect Comparison Charts for {year}\nSolution {solution_id}",
            fontsize=16,
            y=0.98,
        )

        # Add metrics to the overall figure
        if metrics:
            self._add_metrics_to_plot(fig, metrics, solution_id)

        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.85, wspace=0.3, hspace=0.4
        )  # Make room for overall title and metrics

        logger.debug(
            f"Successfully generated quarterly spend effect comparison plots for {year}"
        )
        return fig

    def plot_all(
        self,
        solution_id: str,
        display_plots: bool = True,
        export_location: Union[str, Path] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Create all allocator plots.
        Parameters:
            display_plots (bool): Whether to display the plots
            export_location (Union[str, Path]): Location to export plots
            quiet (bool): If True, suppresses logging output
        """

        try:
            plots = {
                "spend_effect_comparison": self.generate_spend_effect_comparison(
                    solution_id
                ),
                "quarterly_spend_effect_comparison": self.generate_quarterly_spend_effect_comparison(
                    solution_id
                ),
            }

            if display_plots:
                self.display_plots(plots)

            if export_location is not None:
                self.export_plots_fig(export_location, plots)

            return plots

        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise
