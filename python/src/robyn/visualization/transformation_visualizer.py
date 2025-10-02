# pyre-strict
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

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

    @staticmethod
    def _format_metric_value(value: float) -> str:
        """
        Format metric values (CPA/ROAS) with 1 decimal place and space as thousands separator.

        Args:
            value: The numeric value to format

        Returns:
            Formatted string (e.g., "1 234.5" for 1234.567)
        """
        # Round to 1 decimal place
        rounded_value = round(value, 1)

        # Format with thousands separator as space
        return f"{rounded_value:,.1f}".replace(",", " ")

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
        self,
        solution_id: str,
        ax: Optional[plt.Axes] = None,
        filter_channel: Optional[List[str]] = None,
    ) -> Optional[plt.Figure]:
        """Generate comparison plot of spend share vs effect share.

        Args:
            solution_id: ID of the solution to visualize
            ax: Optional matplotlib axes to plot on
            filter_channel: Optional list of channel names to filter out from visualization
        """

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

                # Apply channel filter if provided
                if filter_channel:
                    logger.debug(f"Filtering out channels: {filter_channel}")
                    bar_data = bar_data[~bar_data["rn"].isin(filter_channel)]
                    line_data = line_data[~line_data["rn"].isin(filter_channel)]

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

            # Set up dimensions first to calculate optimal figure size
            # Sort channels by effect share (highest on top, lowest on bottom)
            effect_share_data = bar_data[bar_data["variable"] == "Effect Share"]
            channel_order = (
                effect_share_data.sort_values("value", ascending=True)["rn"]
                .unique()
                .tolist()
            )
            # Use all channels from line_data in case some are missing from bar_data
            channels = [ch for ch in channel_order if ch in line_data["rn"].unique()]
            # Add any missing channels at the bottom
            for ch in line_data["rn"].unique():
                if ch not in channels:
                    channels.append(ch)
            y_pos = np.arange(len(channels))
            num_channels = len(channels)

            logger.debug("Processing %d channels for visualization", num_channels)

            # Calculate dynamic dimensions based on number of channels
            # Minimum height of 8, plus 0.8 for each channel, with a maximum of 20
            dynamic_height = min(max(8, 6 + num_channels * 0.8), 20)

            # Adjust bar width based on number of channels to prevent overlap
            # Use smaller bars when there are many channels
            if num_channels <= 5:
                bar_width = 0.35
            elif num_channels <= 10:
                bar_width = 0.25
            else:
                bar_width = 0.2

            # Adjust font sizes based on number of channels
            if num_channels <= 5:
                label_fontsize = 12
                line_label_fontsize = 12
            elif num_channels <= 10:
                label_fontsize = 11
                line_label_fontsize = 11
            else:
                label_fontsize = 10
                line_label_fontsize = 10

            # Create figure if no axes provided
            if ax is None:
                logger.debug(
                    "Creating new figure and axes with dynamic height: %.1f",
                    dynamic_height,
                )
                fig, ax = plt.subplots(figsize=(16, dynamic_height))
                # Adjust margins based on figure height
                top_margin = max(
                    0.85, 1 - (2.0 / dynamic_height)
                )  # More space for title
                plt.subplots_adjust(top=top_margin, left=0.15, bottom=0.1, right=0.95)
            else:
                logger.debug("Using provided axes for plotting")
                fig = None

            # Set background color
            ax.set_facecolor("white")

            # Set up colors
            type_colour = "#03396C"  # Dark blue for line
            bar_colors = ["#A4C2F4", "#FFB7B2"]  # Light blue and light coral for bars
            bar_colors = bar_colors[::-1]  # Reverse colors

            # Plot bars for each variable type
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
                        fontsize=label_fontsize,
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

            # Recalculate y_sec_scale if channels were filtered to redraw CPA/ROAS line
            if filter_channel and len(line_values) > 0:
                max_bar_value = bar_data["value"].max()
                max_line_value = line_values.max()
                if max_bar_value > 0 and max_line_value > 0:
                    y_sec_scale = max_line_value / max_bar_value * 1.1
                    logger.debug(
                        f"Recalculated y_sec_scale after filtering: {y_sec_scale}"
                    )

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
                    self._format_metric_value(value),
                    color=type_colour,
                    fontweight="bold",
                    fontsize=line_label_fontsize,
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
                y=1.08,
                fontsize=14,
                fontweight="bold",
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
                    label=metric_type,
                )
            ]

            # Combine legend elements in reverse order
            handles = line_legend + list(reversed(bars_legend[0]))
            labels = [metric_type] + list(reversed(bars_legend[1]))

            # Create legend below title with added padding between handle and text
            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(0, 1.05, 0.4, 0.1),
                loc="lower left",
                mode="expand",
                ncol=3,
                frameon=False,
                borderaxespad=0,
                handletextpad=0.5,  # Add padding between legend handle and text
            )

            # Add axis labels
            ax.set_xlabel("Total Share by Channel")
            ax.set_ylabel(None)

            logger.debug("Successfully generated spend effect comparison plot")
            if fig:
                plt.tight_layout()
                # Use the same dynamic top margin calculated earlier
                top_margin = max(0.85, 1 - (2.0 / dynamic_height))
                plt.subplots_adjust(top=top_margin, left=0.15, bottom=0.1, right=0.95)
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

    def _generate_spend_effect_comparison_for_date_range(
        self,
        solution_id: str,
        date_range: List[str],
        ax: Optional[plt.Axes] = None,
        metrics: Optional[Dict[str, float]] = None,
        filter_channel: Optional[List[str]] = None,
    ) -> Union[plt.Figure, bool]:
        """Generate comparison plot of spend share vs effect share for a specific date range.

        Args:
            solution_id: ID of the solution to visualize
            date_range: List of two elements [start_date, end_date] for filtering data
            ax: Optional matplotlib axes to plot on. If None, creates new figure
            metrics: Optional dictionary containing model performance metrics
            filter_channel: Optional list of channel names to filter out from visualization

        Returns:
            Union[plt.Figure, bool]: Generated matplotlib Figure object or True if successful using provided axes
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
            original_bar_data = plot_data["plot1data"]["plotMediaShareLoopBar"].copy()
            original_line_data = plot_data["plot1data"]["plotMediaShareLoopLine"].copy()
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
            return False

        # Get model decomposition data and filter by date
        try:
            # Get data from x_decomp_vec_collect for effects
            x_decomp_vec = self.pareto_result.x_decomp_vec_collect[
                self.pareto_result.x_decomp_vec_collect["sol_id"] == solution_id
            ].copy()

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
                return False

            # Get spend data for the filtered date range
            # First, try to get it from the mediaVecCollect data
            media_vec_collect = self.pareto_result.media_vec_collect[
                (self.pareto_result.media_vec_collect["sol_id"] == solution_id)
                & (self.pareto_result.media_vec_collect["type"] == "rawSpend")
            ].copy()

            # Convert 'ds' to datetime and filter by date range
            if "ds" in media_vec_collect.columns:
                media_vec_collect.loc[:, "ds"] = pd.to_datetime(media_vec_collect["ds"])
                spend_filtered = media_vec_collect[
                    (media_vec_collect["ds"] >= start_date)
                    & (media_vec_collect["ds"] <= end_date)
                ]
            else:
                # Fall back to original data
                spend_filtered = None

            # Calculate effect shares for the filtered date range
            paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends

            # Get effect values for each channel from the filtered data
            effect_values = {}
            total_effect = 0
            for channel in paid_media_spends:
                if channel in date_filtered.columns:
                    effect = date_filtered[channel].sum()
                    effect_values[channel] = effect
                    total_effect += effect

            # Get spend values from the filtered data
            spend_values = {}
            total_spend = 0

            # If we have spend_filtered data, use it to calculate spend shares
            if spend_filtered is not None and not spend_filtered.empty:
                for channel in paid_media_spends:
                    if channel in spend_filtered.columns:
                        spend = spend_filtered[channel].sum()
                        spend_values[channel] = spend
                        total_spend += spend
            else:
                # Fall back to using the same spend shares as in the original data
                for channel in paid_media_spends:
                    channel_data = original_bar_data[original_bar_data["rn"] == channel]
                    if not channel_data.empty:
                        spend_share = channel_data[
                            channel_data["variable"] == "spend_share"
                        ]["value"].iloc[0]
                        # Relative spend share, will be normalized later
                        spend_values[channel] = spend_share
                        total_spend += spend_share

            # Calculate metrics (ROI/CPA)
            # For ROI: effect / spend
            # For CPA: spend / effect
            metric_values = {}
            metric_type = (
                "roi_total"
                if self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
                else "cpa_total"
            )

            for channel in paid_media_spends:
                effect = effect_values.get(channel, 0)
                spend = spend_values.get(channel, 0)

                if metric_type == "roi_total":
                    # ROI = effect / spend
                    metric_values[channel] = effect / spend if spend > 0 else 0
                else:
                    # CPA = spend / effect
                    metric_values[channel] = spend / effect if effect > 0 else 0

            # Create new DataFrames for bar and line data
            bar_data_list = []
            line_data_list = []

            # Get list of channels in proper order
            channels = sorted(paid_media_spends)

            # Convert to shares and create DataFrames
            for channel in channels:
                # Get effect and spend shares
                effect_share = (
                    effect_values.get(channel, 0) / total_effect
                    if total_effect > 0
                    else 0
                )
                spend_share = (
                    spend_values.get(channel, 0) / total_spend if total_spend > 0 else 0
                )

                # Get metric value
                metric_value = metric_values.get(channel, 0)

                # Get original values for nrmse, decomp.rssd, rsq_train from original data
                channel_data = original_bar_data[original_bar_data["rn"] == channel]
                nrmse = channel_data["nrmse"].iloc[0] if not channel_data.empty else 0
                decomp_rssd = (
                    channel_data["decomp.rssd"].iloc[0] if not channel_data.empty else 0
                )
                rsq_train = (
                    channel_data["rsq_train"].iloc[0] if not channel_data.empty else 0
                )

                # Add spend share row FIRST to match order in generate_spend_effect_comparison()
                bar_data_list.append(
                    {
                        "rn": channel,
                        "nrmse": nrmse,
                        "decomp.rssd": decomp_rssd,
                        "rsq_train": rsq_train,
                        "variable": "spend_share",
                        "value": spend_share,
                    }
                )

                # Add effect share row SECOND
                bar_data_list.append(
                    {
                        "rn": channel,
                        "nrmse": nrmse,
                        "decomp.rssd": decomp_rssd,
                        "rsq_train": rsq_train,
                        "variable": "effect_share",
                        "value": effect_share,
                    }
                )

                # Add metric value row
                line_data_list.append(
                    {
                        "rn": channel,
                        "nrmse": nrmse,
                        "decomp.rssd": decomp_rssd,
                        "rsq_train": rsq_train,
                        "variable": metric_type,
                        "value": metric_value,
                    }
                )

            # Create DataFrames from lists
            bar_data = pd.DataFrame(bar_data_list)
            line_data = pd.DataFrame(line_data_list)

            # Apply channel filter if provided
            if filter_channel:
                logger.debug(f"Filtering out channels: {filter_channel}")
                bar_data = bar_data[~bar_data["rn"].isin(filter_channel)]
                line_data = line_data[~line_data["rn"].isin(filter_channel)]

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
            return False

        # Transform variable names
        bar_data["variable"] = bar_data["variable"].str.replace("_", " ").str.title()

        # Set up dimensions first to calculate optimal figure size
        # Sort channels by effect share (highest on top, lowest on bottom)
        effect_share_data = bar_data[bar_data["variable"] == "Effect Share"]
        channel_order = (
            effect_share_data.sort_values("value", ascending=True)["rn"]
            .unique()
            .tolist()
        )
        # Use all channels from line_data in case some are missing from bar_data
        channels = [ch for ch in channel_order if ch in line_data["rn"].unique()]
        # Add any missing channels at the bottom
        for ch in line_data["rn"].unique():
            if ch not in channels:
                channels.append(ch)
        y_pos = np.arange(len(channels))
        num_channels = len(channels)

        # Calculate dynamic dimensions based on number of channels
        # Minimum height of 8, plus 0.8 for each channel, with a maximum of 20
        dynamic_height = min(max(8, 6 + num_channels * 0.8), 20)

        # Adjust bar width based on number of channels to prevent overlap
        # Use smaller bars when there are many channels
        if num_channels <= 5:
            bar_width = 0.35
        elif num_channels <= 10:
            bar_width = 0.25
        else:
            bar_width = 0.2

        # Adjust font sizes based on number of channels
        if num_channels <= 5:
            label_fontsize = 9
            line_label_fontsize = 9
        elif num_channels <= 10:
            label_fontsize = 8
            line_label_fontsize = 8
        else:
            label_fontsize = 7
            line_label_fontsize = 7

        # Create figure if no axes provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(16, dynamic_height))
            # Adjust margins based on figure height
            top_margin = max(0.85, 1 - (2.0 / dynamic_height))  # More space for title
            plt.subplots_adjust(top=top_margin, left=0.15, bottom=0.1, right=0.95)
            return_value = fig
        else:
            fig = None
            return_value = (
                True  # Return True to indicate success when using provided axes
            )

        # Set background color
        ax.set_facecolor("white")

        # Set up colors
        type_colour = "#03396C"  # Dark blue for line
        bar_colors = ["#A4C2F4", "#FFB7B2"]  # Light blue and light coral for bars
        bar_colors = bar_colors[::-1]  # Reverse colors

        # Calculate new y_sec_scale based on the filtered data
        max_bar_value = bar_data["value"].max()
        max_line_value = line_data["value"].max()
        # Cap max_line_value at 50000 to prevent outliers from scaling the chart unreadably
        max_line_value_capped = min(max_line_value, 50000)
        if max_bar_value > 0 and max_line_value_capped > 0:
            y_sec_scale = max_line_value_capped / max_bar_value * 1.1

        # Plot bars for each variable type
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
                    fontsize=label_fontsize,
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

        # Recalculate y_sec_scale if channels were filtered to redraw CPA/ROAS line
        if filter_channel and len(line_values) > 0:
            max_bar_value = bar_data["value"].max()
            max_line_value = line_values.max()
            # Cap max_line_value at 50000 to prevent outliers from scaling the chart unreadably
            max_line_value_capped = min(max_line_value, 50000)
            if max_bar_value > 0 and max_line_value_capped > 0:
                y_sec_scale = max_line_value_capped / max_bar_value * 1.1
                logger.debug(f"Recalculated y_sec_scale after filtering: {y_sec_scale}")

        # Cap line values at 50000 for plotting to keep them within chart boundaries
        # but keep original values for labels
        line_values_capped = np.minimum(line_values, 50000)
        line_x = line_values_capped / y_sec_scale

        # Plot line
        ax.plot(line_x, y_pos, color=type_colour, marker="o", markersize=8, zorder=3)

        # Add line value labels (using original uncapped values)
        for i, value in enumerate(line_values):
            ax.text(
                line_x[i] + 0.02,
                y_pos[i],
                self._format_metric_value(value),
                color=type_colour,
                fontweight="bold",
                fontsize=line_label_fontsize,
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
        metric_type_display = (
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

        # Set title position based on whether this is a standalone or quarterly subplot
        if ax is None or fig is not None:
            # Standalone chart with higher title position
            ax.set_title(
                f"Share of Total Spend, Effect & {metric_type_display}{date_range_str}",
                pad=20,
                y=1.05,
            )
        else:
            # Quarterly subplot with adjusted title position
            ax.set_title(
                f"Share of Total Spend, Effect & {metric_type_display}{date_range_str}",
                pad=30,  # More padding
                y=1.15,  # Higher position
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
                label=metric_type_display,
            )
        ]

        # Combine legend elements in reverse order (FIX HERE)
        handles = line_legend + list(reversed(bars_legend[0]))
        labels = [metric_type_display] + list(reversed(bars_legend[1]))

        # Add legend with better padding between handle and text and adjusted position
        # For quarterly charts (when we're working with provided axes), position the legend lower
        if ax is not None and fig is None:
            # This is a subplot in the quarterly chart
            ax.legend(
                handles=handles,
                labels=labels,
                # Position the legend lower to avoid overlap with the title
                bbox_to_anchor=(0, 0.98, 0.8, 0.1),  # Wider legend area (0.4 -> 0.8)
                loc="lower left",
                mode="expand",
                ncol=3,
                frameon=False,
                borderaxespad=0,
                handletextpad=1.0,  # Increased padding between handle and text
                columnspacing=2.0,  # Added spacing between columns
            )
        else:
            # This is a standalone chart
            ax.legend(
                handles=handles,
                labels=labels,
                bbox_to_anchor=(0, 1.05, 0.4, 0.1),
                loc="lower left",
                mode="expand",
                ncol=3,
                frameon=False,
                borderaxespad=0,
                handletextpad=0.5,  # Add padding between legend handle and text
            )

        # Add axis labels
        ax.set_xlabel("Total Share by Channel")
        ax.set_ylabel(None)

        logger.debug(
            "Successfully generated date-filtered spend effect comparison plot"
        )
        return return_value

    def _get_last_four_quarters(self) -> List[List[str]]:
        """
        Get date ranges for the last four quarters counting backward from the latest date.

        Returns:
            List of four date ranges, each containing [start_date, end_date] as strings.
            The quarters are ordered from oldest to newest.
        """
        # Get the available dates from the data
        ts_data = None

        # Try to get dates from decomp_vec if available
        if (
            hasattr(self.pareto_result, "x_decomp_vec_collect")
            and not self.pareto_result.x_decomp_vec_collect.empty
        ):
            ts_data = pd.to_datetime(
                self.pareto_result.x_decomp_vec_collect["ds"].unique()
            )
        elif (
            self.mmm_data
            and hasattr(self.mmm_data, "dt")
            and "ds" in self.mmm_data.dt.columns
        ):
            # Get dates directly from mmm_data
            ts_data = pd.to_datetime(self.mmm_data.dt["ds"].unique())

        if ts_data is None or len(ts_data) == 0:
            logger.warning(
                "Could not find date information to determine last four quarters."
            )
            # Return default dates (fallback to today)
            ref_date = pd.Timestamp.now()
        else:
            # Find the latest date in the data and add 7 days (to account for the week ending)
            ref_date = pd.Timestamp(max(ts_data)) + pd.Timedelta(days=7)

        # Calculate current quarter information
        current_year = ref_date.year
        current_quarter = (ref_date.month - 1) // 3 + 1

        # Generate the last four quarters
        quarters = []
        for i in range(4):
            # Calculate quarter and year (counting backward)
            quarter = current_quarter - i
            year = current_year

            # Adjust for previous year if needed
            while quarter <= 0:
                quarter += 4
                year -= 1

            # Calculate start and end months
            start_month = (quarter - 1) * 3 + 1
            end_month = quarter * 3

            # Create start and end dates
            start_date = pd.Timestamp(f"{year}-{start_month:02d}-01")

            # End date is the last day of the end month
            if end_month == 12:
                end_date = pd.Timestamp(f"{year}-12-31")
            else:
                next_month_year = year
                next_month = end_month + 1
                if next_month > 12:
                    next_month = 1
                    next_month_year += 1
                end_date = pd.Timestamp(
                    f"{next_month_year}-{next_month:02d}-01"
                ) - pd.Timedelta(days=1)

            # For the most recent quarter (i==0), cap at the reference date
            if i == 0 and end_date > ref_date:
                end_date = ref_date

            # Add to list
            quarters.append(
                [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            )

        # Return quarters from oldest to newest
        return sorted(quarters, key=lambda x: x[0])

    def generate_quarterly_spend_effect_comparison(
        self,
        solution_id: str,
        metrics: Optional[Dict[str, float]] = None,
        filter_channel: Optional[List[str]] = None,
    ) -> Optional[plt.Figure]:
        """
        Generate spend effect comparison charts for each of the last four quarters in a 2x2 grid.

        Args:
            solution_id: ID of the solution to visualize
            metrics: Optional dictionary containing model performance metrics
            filter_channel: Optional list of channel names to filter out from visualization

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

        # Get last four quarters date ranges (oldest to newest)
        quarters = self._get_last_four_quarters()

        # Create quarter names based on the quarter dates
        quarter_names = []
        for quarter_range in quarters:
            start_date = pd.to_datetime(quarter_range[0])
            quarter_number = (start_date.month - 1) // 3 + 1
            quarter_names.append(f"Q{quarter_number} {start_date.year}")

        # Create a figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()

        # Generate spend effect comparison for each quarter
        for i, (quarter_range, quarter_name) in enumerate(zip(quarters, quarter_names)):
            try:
                # Call method for date-filtered spend effect comparison
                quarter_plot_success = (
                    self._generate_spend_effect_comparison_for_date_range(
                        solution_id=solution_id,
                        date_range=quarter_range,
                        ax=axes[i],
                        metrics=None,  # Don't add metrics to individual subplots
                        filter_channel=filter_channel,
                    )
                )

                # If the plot generation failed, display an error message
                if quarter_plot_success is not True:
                    axes[i].text(
                        0.5,
                        0.5,
                        f"No data available for {quarter_name}",
                        ha="center",
                        va="center",
                        transform=axes[i].transAxes,
                    )

                # Set simplified title for quarterly charts
                axes[i].set_title(f"{quarter_name}", pad=20, y=1.15)

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
        metric_type_display = (
            "ROAS"
            if (
                self.mmm_data
                and hasattr(self.mmm_data.mmmdata_spec, "dep_var_type")
                and self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
            )
            else "CPA"
        )

        # Get date range for title
        start_date = pd.to_datetime(quarters[0][0])
        end_date = pd.to_datetime(quarters[-1][1])
        date_range = (
            f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        fig.suptitle(
            f"Quarterly Spend & Effect Share Comparison with {metric_type_display} ({date_range})\nSolution {solution_id}",
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
            f"Successfully generated quarterly spend effect comparison plots for last four quarters"
        )
        return fig

    def dump_monthly_spend_effect_csv(
        self,
        solution_id: str,
        filepath: Union[str, Path] = None,
        export_raw_x_decomp: bool = True,
    ) -> Optional[Dict[str, str]]:
        """
        Dump monthly spend effect comparison data to CSV file.

        Args:
            solution_id: ID of the solution to extract data from
            filepath: Optional path to save the CSV file. If None, saves to current directory
            export_raw_x_decomp: Whether to also export raw x_decomp_vec data to separate CSV

        Returns:
            Optional[Dict[str, str]]: Dict with paths to saved CSV files, or None if failed
                - 'monthly_summary': Path to monthly aggregated summary CSV
                - 'raw_x_decomp': Path to raw x_decomp_vec CSV (if export_raw_x_decomp=True)
        """
        logger.debug("Starting generation of monthly spend effect CSV dump")

        try:
            # Check if solution_id exists in the data
            if solution_id not in self.pareto_result.plot_data_collect:
                logger.warning(
                    f"Invalid solution ID: {solution_id}. Solution not found in available data."
                )
                return None

            # Get decomposition data for effects
            x_decomp_vec = self.pareto_result.x_decomp_vec_collect[
                self.pareto_result.x_decomp_vec_collect["sol_id"] == solution_id
            ].copy()

            if x_decomp_vec.empty:
                logger.warning(
                    f"No decomposition data found for solution {solution_id}"
                )
                return None

            # Convert 'ds' to datetime and extract month
            x_decomp_vec.loc[:, "ds"] = pd.to_datetime(x_decomp_vec["ds"])
            x_decomp_vec.loc[:, "month"] = x_decomp_vec["ds"].dt.to_period("M")

            # Get spend data
            media_vec_collect = self.pareto_result.media_vec_collect[
                (self.pareto_result.media_vec_collect["sol_id"] == solution_id)
                & (self.pareto_result.media_vec_collect["type"] == "rawSpend")
            ].copy()

            # If spend data has dates, add month column
            if "ds" in media_vec_collect.columns and not media_vec_collect.empty:
                media_vec_collect.loc[:, "ds"] = pd.to_datetime(media_vec_collect["ds"])
                media_vec_collect.loc[:, "month"] = media_vec_collect[
                    "ds"
                ].dt.to_period("M")

            # Get channel names
            paid_media_spends = self.mmm_data.mmmdata_spec.paid_media_spends

            # Determine metric type
            metric_type = (
                "ROAS"
                if self.mmm_data.mmmdata_spec.dep_var_type == DependentVarType.REVENUE
                else "CPA"
            )

            # Prepare CSV data
            csv_data = []

            # Get unique months from decomposition data
            months = sorted(x_decomp_vec["month"].unique())

            for month in months:
                month_str = str(month)

                # Filter data for this month
                month_decomp = x_decomp_vec[x_decomp_vec["month"] == month]

                # Calculate monthly totals for effects
                total_monthly_effect = 0
                monthly_effects = {}
                for channel in paid_media_spends:
                    if channel in month_decomp.columns:
                        effect = month_decomp[channel].sum()
                        monthly_effects[channel] = effect
                        total_monthly_effect += effect

                # Calculate monthly totals for spend
                total_monthly_spend = 0
                monthly_spends = {}

                if not media_vec_collect.empty and "month" in media_vec_collect.columns:
                    month_spend = media_vec_collect[media_vec_collect["month"] == month]
                    for channel in paid_media_spends:
                        if channel in month_spend.columns:
                            spend = month_spend[channel].sum()
                            monthly_spends[channel] = spend
                            total_monthly_spend += spend
                else:
                    # Fallback: use overall spend shares proportionally
                    plot_data = self.pareto_result.plot_data_collect[solution_id]
                    original_bar_data = plot_data["plot1data"][
                        "plotMediaShareLoopBar"
                    ].copy()

                    # Get total effect for this month as proxy for relative spend
                    for channel in paid_media_spends:
                        channel_data = original_bar_data[
                            original_bar_data["rn"] == channel
                        ]
                        if not channel_data.empty:
                            spend_share = channel_data[
                                channel_data["variable"] == "spend_share"
                            ]["value"].iloc[0]
                            # Estimate spend based on spend share and monthly effect proportion
                            estimated_spend = spend_share * total_monthly_effect
                            monthly_spends[channel] = estimated_spend
                            total_monthly_spend += estimated_spend

                # Create CSV rows for each channel
                for channel in paid_media_spends:
                    effect = monthly_effects.get(channel, 0)
                    spend = monthly_spends.get(channel, 0)

                    # Calculate shares
                    effect_share = (
                        effect / total_monthly_effect if total_monthly_effect > 0 else 0
                    )
                    spend_share = (
                        spend / total_monthly_spend if total_monthly_spend > 0 else 0
                    )

                    # Calculate CPA/ROAS
                    if metric_type == "ROAS":
                        cpa_roas = effect / spend if spend > 0 else 0
                    else:
                        cpa_roas = spend / effect if effect > 0 else 0

                    csv_data.append(
                        {
                            "month": month_str,
                            "spend_name": channel,
                            "effect_share": round(effect_share, 4),
                            "spend_share": round(spend_share, 4),
                            "CPA": round(cpa_roas, 4),
                        }
                    )

            # Set default filepath if not provided
            result_paths = {}

            if filepath is None:
                summary_filepath = f"monthly_spend_effect_comparison_{solution_id}.csv"
                raw_filepath = f"raw_x_decomp_vec_{solution_id}.csv"
            else:
                filepath = Path(filepath)
                if filepath.is_dir():
                    summary_filepath = (
                        filepath / f"monthly_spend_effect_comparison_{solution_id}.csv"
                    )
                    raw_filepath = filepath / f"raw_x_decomp_vec_{solution_id}.csv"
                else:
                    # filepath is a file - use its directory and stem for naming
                    base_name = filepath.stem
                    directory = filepath.parent
                    summary_filepath = directory / f"{base_name}_monthly_summary.csv"
                    raw_filepath = directory / f"{base_name}_raw_x_decomp.csv"

            # Write monthly summary CSV
            with open(summary_filepath, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = [
                    "month",
                    "spend_name",
                    "effect_share",
                    "spend_share",
                    "CPA",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                writer.writeheader()
                for row in csv_data:
                    writer.writerow(row)

            logger.info(
                f"Successfully saved monthly spend effect CSV to: {summary_filepath}"
            )
            result_paths["monthly_summary"] = str(summary_filepath)

            # Export raw x_decomp_vec data if requested
            if export_raw_x_decomp:
                try:
                    x_decomp_vec.to_csv(raw_filepath, index=False)
                    logger.info(
                        f"Successfully saved raw x_decomp_vec CSV to: {raw_filepath}"
                    )
                    result_paths["raw_x_decomp"] = str(raw_filepath)
                except Exception as e:
                    logger.warning(f"Failed to save raw x_decomp_vec CSV: {str(e)}")

            return result_paths

        except Exception as e:
            logger.error(f"Failed to generate monthly spend effect CSV: {str(e)}")
            return None

    def plot_all(
        self,
        solution_id: str,
        display_plots: bool = True,
        export_location: Union[str, Path] = None,
        filter_channel: Optional[List[str]] = None,
    ) -> Dict[str, plt.Figure]:
        """
        Create all allocator plots and generate monthly spend effect CSV.
        Parameters:
            display_plots (bool): Whether to display the plots
            export_location (Union[str, Path]): Location to export plots and CSV
            filter_channel (Optional[List[str]]): List of channel names to filter out from visualization.
                The numbers on the plot remain unchanged, but the CPA/ROAS line is redrawn after filtering.
                Useful for beautifying charts when there are outliers with very high CPA numbers.
        """

        try:
            plots = {
                "spend_effect_comparison": self.generate_spend_effect_comparison(
                    solution_id, filter_channel=filter_channel
                ),
                "quarterly_spend_effect_comparison": self.generate_quarterly_spend_effect_comparison(
                    solution_id, filter_channel=filter_channel
                ),
            }

            # Generate monthly spend effect CSV
            csv_export_path = export_location if export_location is not None else None
            csv_file_paths = self.dump_monthly_spend_effect_csv(
                solution_id=solution_id, filepath=csv_export_path
            )

            if csv_file_paths:
                for file_type, file_path in csv_file_paths.items():
                    logger.info(
                        f"{file_type.replace('_', ' ').title()} CSV saved to: {file_path}"
                    )
            else:
                logger.warning("Failed to generate monthly spend effect CSV")

            if display_plots:
                self.display_plots(plots)

            if export_location is not None:
                self.export_plots_fig(export_location, plots)

            return plots

        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise
