from pathlib import Path
from typing import Optional, Union, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import logging
from robyn.data.entities.mmmdata import MMMData
from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class ResponseVisualizer(BaseVisualizer):
    def __init__(self, pareto_result: ParetoResult, mmm_data: MMMData):
        super().__init__()
        logger.debug(
            "Initializing ResponseVisualizer with pareto_result=%s, mmm_data=%s",
            pareto_result,
            mmm_data,
        )
        self.pareto_result = pareto_result
        self.mmm_data = mmm_data

    def plot_response(self) -> plt.Figure:
        """
        Plot response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting response curve plotting")

        # Get the first solution ID from the pareto result
        solution_ids = list(self.pareto_result.plot_data_collect.keys())
        if not solution_ids:
            logger.error("No solution IDs found in pareto result")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No solution IDs found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Use the first solution ID
        solution_id = solution_ids[0]
        logger.info(f"Using solution ID: {solution_id}")

        # Generate the response curves
        fig = self.generate_response_curves(solution_id)

        # If generation failed, create an empty figure with an error message
        if fig is None:
            logger.error("Failed to generate response curves")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Failed to generate response curves",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        return fig

    def plot_marginal_response(self) -> plt.Figure:
        """
        Plot marginal response curves.

        Returns:
            plt.Figure: The generated figure.
        """
        logger.info("Starting marginal response curve plotting")

        # Create a figure with multiple subplots, one for each solution
        solution_ids = list(self.pareto_result.plot_data_collect.keys())
        if not solution_ids:
            logger.error("No solution IDs found in pareto result")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "No solution IDs found",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        # Create a figure with subplots
        fig, axes = plt.subplots(
            len(solution_ids), 1, figsize=(10, 6 * len(solution_ids))
        )

        # If there's only one solution, axes will be a single Axes object, not an array
        if len(solution_ids) == 1:
            axes = [axes]

        # Generate response curves for each solution
        for i, solution_id in enumerate(solution_ids):
            logger.info(f"Generating response curves for solution ID: {solution_id}")
            self.generate_response_curves(solution_id, ax=axes[i])
            axes[i].set_title(f"Solution ID: {solution_id}")

        # Adjust layout
        fig.tight_layout()

        return fig

    def generate_response_curves(
        self, solution_id: str, ax: Optional[plt.Axes] = None, trim_rate: float = 1.3
    ) -> Optional[plt.Figure]:
        """Generate response curves showing relationship between spend and response with improved stability."""
        logger.debug("Generating response curves with trim_rate=%.2f", trim_rate)

        if solution_id not in self.pareto_result.plot_data_collect:
            logger.error(f"Invalid solution ID: {solution_id}")
            if ax is not None:
                ax.text(
                    0.5,
                    0.5,
                    f"Invalid solution ID: {solution_id}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return None

        try:
            # Add debug logging to inspect plot data structure
            plot_data = self.pareto_result.plot_data_collect[solution_id]

            # Log plot data keys
            logger.debug(f"plot_data keys: {list(plot_data.keys())}")

            # Check if required data exists
            if (
                "plot4data" not in plot_data
                or "dt_scurvePlot" not in plot_data["plot4data"]
                or "dt_scurvePlotMean" not in plot_data["plot4data"]
            ):
                logger.error(f"Missing required data for solution {solution_id}")
                if ax is not None:
                    ax.text(
                        0.5,
                        0.5,
                        "Missing required data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None

            curve_data = plot_data["plot4data"]["dt_scurvePlot"].copy()
            mean_data = plot_data["plot4data"]["dt_scurvePlotMean"].copy()

            # Log data shapes and unique channel names for debugging
            logger.debug(f"curve_data shape: {curve_data.shape}")
            logger.debug(f"mean_data shape: {mean_data.shape}")
            logger.debug(
                f"Unique channels in curve_data: {curve_data['channel'].unique()}"
            )
            logger.debug(
                f"Unique channels in mean_data: {mean_data['channel'].unique()}"
            )

            # Early safety check for empty dataframes
            if curve_data.empty or mean_data.empty:
                logger.error(f"Empty data for solution {solution_id}")
                if ax is not None:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None

            # Add filtering for paid media channels and trim rate
            try:
                # Check for paid media channels
                paid_media_channels = self.mmm_data.mmmdata_spec.paid_media_spends
                if not paid_media_channels:
                    logger.warning("No paid media channels found")

                logger.debug(f"Paid media channels: {paid_media_channels}")

                # FIX: Check if any channels in curve_data match paid_media_channels
                matching_channels = set(curve_data["channel"].unique()).intersection(
                    paid_media_channels
                )
                if not matching_channels:
                    logger.warning(
                        f"None of the channels in curve_data match paid_media_channels. Showing all channels."
                    )
                else:
                    # Filter to only include paid media channels
                    curve_data = curve_data[
                        curve_data["channel"].isin(paid_media_channels)
                    ]
                    logger.debug(
                        f"After filtering, curve_data shape: {curve_data.shape}"
                    )
                    logger.debug(
                        f"After filtering, unique channels in curve_data: {curve_data['channel'].unique()}"
                    )

                # Check if we still have data after filtering
                if curve_data.empty:
                    logger.error("No data left after filtering for paid media channels")
                    if ax is not None:
                        ax.text(
                            0.5,
                            0.5,
                            "No data for paid media channels",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    return None

                # Apply trim rate if provided
                if trim_rate > 0:
                    # Calculate maximum values, with safeguards against empty data
                    if not mean_data.empty:
                        # FIX: Use more robust approach to calculate max values
                        # Use 95th percentile instead of max to avoid extreme outliers
                        if len(mean_data["mean_spend_adstocked"]) > 0:
                            max_mean_spend = np.nanpercentile(
                                mean_data["mean_spend_adstocked"]
                                .replace([np.inf, -np.inf], np.nan)
                                .dropna(),
                                95,
                            )
                        else:
                            max_mean_spend = float("inf")

                        if len(mean_data["mean_response"]) > 0:
                            max_mean_response = np.nanpercentile(
                                mean_data["mean_response"]
                                .replace([np.inf, -np.inf], np.nan)
                                .dropna(),
                                95,
                            )
                        else:
                            max_mean_response = float("inf")

                        logger.debug(f"Calculated max_mean_spend: {max_mean_spend}")
                        logger.debug(
                            f"Calculated max_mean_response: {max_mean_response}"
                        )

                        # Add safety check for max values
                        if np.isfinite(max_mean_spend) and np.isfinite(
                            max_mean_response
                        ):
                            # FIX: Make trim filtering more lenient to prevent removing all data
                            filtered_curve_data = curve_data[
                                (curve_data["spend"] < max_mean_spend * trim_rate)
                                & (
                                    curve_data["response"]
                                    < max_mean_response * trim_rate
                                )
                                & (
                                    curve_data["channel"].isin(
                                        matching_channels
                                        or curve_data["channel"].unique()
                                    )
                                )
                            ]

                            # Only apply filtering if we don't lose all data
                            if not filtered_curve_data.empty:
                                curve_data = filtered_curve_data
                                logger.debug(
                                    f"After trim, curve_data shape: {curve_data.shape}"
                                )
                            else:
                                logger.warning(
                                    "Trim filtering would remove all data points. Skipping trim."
                                )
                        else:
                            logger.warning(
                                "Found non-finite maximum values, skipping trim"
                            )

                # Filter mean data to match curve data channels
                mean_data = mean_data[
                    mean_data["channel"].isin(curve_data["channel"].unique())
                ]
                logger.debug(f"After filtering, mean_data shape: {mean_data.shape}")

                # Check if we still have data
                if curve_data.empty or mean_data.empty:
                    logger.error("No data left after trimming")
                    if ax is not None:
                        ax.text(
                            0.5,
                            0.5,
                            "Insufficient data after trimming",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    return None
            except Exception as filter_error:
                logger.error(f"Error during data filtering: {str(filter_error)}")
                if ax is not None:
                    ax.text(
                        0.5,
                        0.5,
                        "Error filtering data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None

            # Replace inf or extremely large values with NaN
            try:
                # Replace inf or extremely large values with NaN
                curve_data = curve_data.replace([np.inf, -np.inf], np.nan)
                mean_data = mean_data.replace([np.inf, -np.inf], np.nan)

                # Drop rows with NaN values
                curve_data = curve_data.dropna(subset=["spend", "response"])
                mean_data = mean_data.dropna(
                    subset=["mean_spend_adstocked", "mean_response"]
                )

                # Add safety check again after processing
                curve_data = curve_data.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["spend", "response"]
                )
                mean_data = mean_data.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=["mean_spend_adstocked", "mean_response"]
                )

                # Check if we still have data
                if curve_data.empty or mean_data.empty:
                    logger.error("No valid data after processing")
                    if ax is not None:
                        ax.text(
                            0.5,
                            0.5,
                            "No valid data after processing",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                        )
                    return None
            except Exception as processing_error:
                logger.error(f"Error during data processing: {str(processing_error)}")
                if ax is not None:
                    ax.text(
                        0.5,
                        0.5,
                        "Error processing data",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None

            # Create new axes if needed
            if ax is None:
                logger.debug("Creating new figure with axes")
                fig, ax = plt.subplots(figsize=(16, 10))
            else:
                logger.debug("Using provided axes")
                fig = None

            # Generate dynamic color map based on actual channel names
            def get_channel_colors(channels: List[str]) -> Dict[str, str]:
                base_colors = [
                    "#FF9D1C",  # Orange
                    "#69B3E7",  # Light Blue
                    "#7B4EA3",  # Purple
                    "#E41A1C",  # Red
                    "#4DAF4A",  # Green
                    "#377EB8",  # Blue
                    "#984EA3",  # Violet
                    "#FF7F00",  # Dark Orange
                    "#FFFF33",  # Yellow
                    "#A65628",  # Brown
                    "#F781BF",  # Pink
                    "#999999",  # Gray
                    "#1CE9F4",  # Teal
                    "#E3256B",  # Magenta
                    "#006400",  # Dark Green
                    "#FF7F50",  # Coral
                    "#4B0082",  # Indigo
                    "#FFD700",  # Gold
                    "#4682B4",  # Steel Blue
                    "#FA8072",  # Salmon
                ]
                color_mapping = {}
                for i, channel in enumerate(channels):
                    color_mapping[channel] = base_colors[i % len(base_colors)]
                return color_mapping

            try:
                # Get list of channels
                channels = curve_data["channel"].unique()
                logger.debug(
                    "Processing %d unique channels: %s", len(channels), channels
                )

                # Generate dynamic color map instead of using hardcoded one
                color_map = get_channel_colors(channels)
                logger.debug(f"Generated color map: {color_map}")

                # Set dynamic axis limits based on data
                max_x = (
                    max(
                        curve_data["spend"].max(),
                        mean_data["mean_spend_adstocked"].max(),
                    )
                    * 1.1
                )
                max_y = (
                    max(curve_data["response"].max(), mean_data["mean_response"].max())
                    * 1.1
                )

                # Use fallback values if needed, but not fixed limits
                max_x = max(max_x, 100) if np.isfinite(max_x) else 100
                max_y = max(max_y, 100) if np.isfinite(max_y) else 100

                # Make sure limits are reasonable (cap to prevent memory issues with very large values)
                max_x = min(max_x, 1000)
                max_y = min(max_y, 1000)

                # Plot each channel response curve
                for channel in channels:
                    logger.debug("Plotting response curve for channel: %s", channel)
                    channel_data = curve_data[
                        curve_data["channel"] == channel
                    ].sort_values("spend")

                    # Skip if too few data points
                    if len(channel_data) < 2:
                        logger.warning(f"Not enough data points for channel {channel}")
                        continue

                    # Get color from our dynamic map
                    color = color_map.get(
                        channel, "gray"
                    )  # Default to gray if channel not in map

                    # Plot curve
                    ax.plot(
                        channel_data["spend"],
                        channel_data["response"],
                        color=color,
                        label=channel,
                        zorder=2,
                    )

                    # Add shaded area with simpler approach to avoid memory issues
                    try:
                        if "mean_carryover" in mean_data.columns:
                            channel_mean = mean_data[mean_data["channel"] == channel]
                            if not channel_mean.empty and np.isfinite(
                                channel_mean["mean_carryover"].iloc[0]
                            ):
                                mean_carryover = channel_mean["mean_carryover"].iloc[
                                    0
                                ]  # No scaling needed

                                # Simplified shading - use fewer points to reduce memory usage
                                if mean_carryover > 0:
                                    # Get subset of points for shading (no more than 50 points)
                                    shading_data = channel_data[
                                        channel_data["spend"] <= mean_carryover
                                    ]
                                    if len(shading_data) > 50:
                                        indices = np.linspace(
                                            0, len(shading_data) - 1, 50, dtype=int
                                        )
                                        shading_data = shading_data.iloc[indices]

                                    ax.fill_between(
                                        shading_data["spend"],
                                        np.zeros_like(shading_data["spend"]),
                                        shading_data["response"],
                                        color="grey",
                                        alpha=0.3,  # Reduced alpha for better performance
                                        zorder=1,
                                    )
                    except Exception as shading_error:
                        logger.warning(
                            f"Error adding shading for {channel}: {str(shading_error)}"
                        )
                        # Continue with the plot even if shading fails
                        pass

                # Add mean points with simplified approach
                try:
                    for idx, row in mean_data.iterrows():
                        channel = row["channel"]
                        mean_spend = row["mean_spend_adstocked"]
                        mean_response = row["mean_response"]

                        if not (np.isfinite(mean_spend) and np.isfinite(mean_response)):
                            continue

                        color = color_map.get(channel, "gray")

                        # Only add if within axis limits
                        if mean_spend <= max_x and mean_response <= max_y:
                            ax.scatter(
                                mean_spend,
                                mean_response,
                                color=color,
                                s=80,  # Reduced size from original
                                zorder=3,
                            )

                            # Only add labels for important points to reduce clutter
                            if (
                                mean_response > max_y * 0.2
                            ):  # Only label points above 20% of max
                                # Format without K suffix since we're not scaling
                                formatted_spend = f"{mean_spend:.1f}"

                                ax.text(
                                    mean_spend + max_x * 0.02,  # Small offset
                                    mean_response,
                                    formatted_spend,
                                    ha="left",
                                    va="bottom",
                                    fontsize=8,  # Smaller font
                                    color=color,
                                )
                except Exception as point_error:
                    logger.warning(f"Error adding mean points: {str(point_error)}")
                    # Continue even if point labels fail
                    pass

                # Format axes - simplified from original to improve performance
                logger.debug("Formatting axis labels")

                # Custom tick formatter with error handling - no K suffix since values are not scaled
                def custom_tick_formatter(x, p):
                    try:
                        if x == 0:
                            return "0"
                        if abs(x) < 0.01:
                            return (
                                f"{x:.2e}"  # Scientific notation for very small numbers
                            )
                        elif abs(x) < 1:
                            return f"{x:.2f}"
                        elif abs(x) < 1000:
                            return f"{int(x)}" if x.is_integer() else f"{x:.1f}"
                        else:
                            # For large numbers, use K/M/B notation
                            if abs(x) < 1_000_000:
                                return f"{x/1000:.1f}K"
                            elif abs(x) < 1_000_000_000:
                                return f"{x/1_000_000:.1f}M"
                            else:
                                return f"{x/1_000_000_000:.1f}B"
                    except:
                        return str(x)

                # Set axis limits dynamically
                ax.set_xlim(0, max_x)
                ax.set_ylim(0, max_y)

                # Set major ticks dynamically but with fewer ticks to reduce complexity
                try:
                    x_ticks = np.linspace(0, max_x, min(5, int(max_x / 20) + 1))
                    y_ticks = np.linspace(0, max_y, min(6, int(max_y / 20) + 1))

                    # Round to nice values
                    x_ticks = np.round(x_ticks, 1)
                    y_ticks = np.round(y_ticks, 1)

                    ax.set_xticks(x_ticks)
                    ax.set_yticks(y_ticks)

                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(custom_tick_formatter)
                    )
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(custom_tick_formatter)
                    )
                except Exception as tick_error:
                    logger.warning(f"Error setting tick marks: {str(tick_error)}")
                    # Fall back to automatic ticks
                    pass

                # Simplified grid styling
                ax.grid(True, alpha=0.2, linestyle="-", color="gray")
                ax.set_axisbelow(True)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                # Set titles
                ax.set_title("Response Curves and Mean Spends by Channel")
                ax.set_xlabel("Spend (carryover + immediate)")
                ax.set_ylabel("Response")

                # Simplify legend to prevent memory issues
                try:
                    # Limit to 8 channels maximum to prevent legend overflow
                    if len(channels) > 8:
                        top_channels = (
                            mean_data.sort_values("mean_response", ascending=False)[
                                "channel"
                            ]
                            .head(8)
                            .tolist()
                        )
                        handles, labels = ax.get_legend_handles_labels()

                        # Keep only top channels
                        included_handles = [
                            h for h, l in zip(handles, labels) if l in top_channels
                        ]
                        included_labels = [l for l in labels if l in top_channels]

                        # Simplified legend
                        ax.legend(
                            included_handles,
                            included_labels,
                            loc="upper left",
                            frameon=True,
                            framealpha=0.7,
                            fontsize=8,  # Smaller font
                        )
                    else:
                        ax.legend(
                            loc="upper left",
                            frameon=True,
                            framealpha=0.7,
                            fontsize=8,  # Smaller font
                        )
                except Exception as legend_error:
                    logger.warning(f"Error creating legend: {str(legend_error)}")
                    # Try minimal legend as fallback
                    try:
                        ax.legend(loc="best")
                    except:
                        pass

                # Release memory explicitly for large objects no longer needed
                del curve_data, mean_data, channels

                if fig:
                    logger.debug("Adjusting layout")
                    fig.tight_layout()
                    logger.debug("Successfully generated response curves figure")
                    return fig

                logger.debug("Successfully added response curves to existing axes")
                return None

            except Exception as plotting_error:
                logger.error(f"Error during plotting: {str(plotting_error)}")
                # If we have an axis, show error
                if ax is not None:
                    ax.clear()  # Clear the axis completely
                    ax.text(
                        0.5,
                        0.5,
                        "Error plotting response curves",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                return None

        except Exception as e:
            logger.error("Error generating response curves: %s", str(e), exc_info=True)
            if ax is not None:
                ax.clear()  # Clear the axis completely
                ax.text(
                    0.5,
                    0.5,
                    "Error generating response curves",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            return None

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> None:
        """
        Generate and optionally display or export all plots.

        Args:
            display_plots (bool, optional): Whether to display the plots. Defaults to True.
            export_location (Union[str, Path], optional): Path to export the plots. Defaults to None.
        """
        logger.info("Generating all plots")

        # Create a dictionary to hold the figures
        figures = {}

        # Generate response curves
        try:
            figures["response"] = self.plot_response()
        except Exception as e:
            logger.error(f"Error generating response plot: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Error generating response plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            figures["response"] = fig

        # Generate marginal response curves
        try:
            figures["marginal_response"] = self.plot_marginal_response()
        except Exception as e:
            logger.error(f"Error generating marginal response plot: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(
                0.5,
                0.5,
                "Error generating marginal response plot",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            figures["marginal_response"] = fig

        # Display plots if requested
        if display_plots:
            for name, fig in figures.items():
                plt.figure(fig.number)
                plt.show()

        # Export plots if a location is provided
        if export_location:
            export_path = Path(export_location)
            export_path.mkdir(parents=True, exist_ok=True)

            for name, fig in figures.items():
                file_path = export_path / f"{name}.png"
                fig.savefig(file_path, bbox_inches="tight")
                logger.info(f"Exported {name} plot to {file_path}")
