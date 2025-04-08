import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Union
from pathlib import Path
import logging

from robyn.allocator.entities.allocation_result import (
    AllocationResult,
)
from robyn.allocator.optimizer import BudgetAllocator
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class AllocatorPlotter(BaseVisualizer):
    """Generates plots for Robyn allocator results."""

    def __init__(
        self,
        allocation_result: AllocationResult,
        budget_allocator: BudgetAllocator,
    ):
        """
        Initialize AllocatorPlotter with all necessary data.

        Args:
            allocation_result: Results from budget allocation
            budget_allocator: Budget allocator instance with model parameters
        """
        super().__init__()
        logger.info("Initializing AllocatorPlotter")

        # Store provided data
        self.allocation_result = allocation_result
        self.budget_allocator = budget_allocator

        # Store commonly used data
        self.dt_optimOut = allocation_result.dt_optimOut
        self.mainPoints = allocation_result.mainPoints

        # Infer plotting configurations
        self.metric = (
            "ROAS"
            if budget_allocator.mmm_data.mmmdata_spec.dep_var_type == "revenue"
            else "CPA"
        )
        self.interval_type = budget_allocator.mmm_data.mmmdata_spec.interval_type
        self.model_id = budget_allocator.select_model
        self.scenario = allocation_result.scenario

        # Pre-calculate scenario metrics
        self.scenarios = {
            "Initial": {
                "spend": self.dt_optimOut.init_spend_unit.sum(),
                "response": self.dt_optimOut.init_response_unit.sum(),
            },
            "Bounded": {
                "spend": self.dt_optimOut.optm_spend_unit.sum(),
                "response": self.dt_optimOut.optm_response_unit.sum(),
            },
            "Bounded x3": {
                "spend": self.dt_optimOut.optm_spend_unit_unbound.sum(),
                "response": self.dt_optimOut.optm_response_unit_unbound.sum(),
            },
        }

        # Calculate percentage changes and metrics for each scenario
        for scenario in self.scenarios:
            if scenario != "Initial":
                self.scenarios[scenario].update(
                    {
                        "spend_pct_change": (
                            self.scenarios[scenario]["spend"]
                            / self.scenarios["Initial"]["spend"]
                            - 1
                        )
                        * 100,
                        "response_pct_change": (
                            self.scenarios[scenario]["response"]
                            / self.scenarios["Initial"]["response"]
                            - 1
                        )
                        * 100,
                    }
                )
                if self.metric == "ROAS":
                    self.scenarios[scenario]["metric_value"] = (
                        self.scenarios[scenario]["response"]
                        / self.scenarios[scenario]["spend"]
                    )
                else:  # CPA
                    self.scenarios[scenario]["metric_value"] = (
                        self.scenarios[scenario]["spend"]
                        / self.scenarios[scenario]["response"]
                    )

        logger.debug("AllocatorPlotter initialized with data")

    def _plot_allocation_matrix(self) -> plt.Figure:
        """Plot allocation matrix as three adjacent heatmaps with no space between them."""
        if self.dt_optimOut is None:
            raise ValueError("dt_optimOut must be provided")

        logger.info("Creating allocation matrix plot")
        try:
            # Create figure with GridSpec for precise layout control (no spaces)
            fig = plt.figure(figsize=(15, 6))
            gs = fig.add_gridspec(1, 3, wspace=0)  # Zero spacing between columns
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

            scenarios = ["Initial", "Bounded", "Bounded x3"]
            # Remove mCPA column from metrics
            metrics = [
                "abs.mean\nspend",
                "mean\nspend%",
                "mean\nresponse%",
                f"mean\n{self.metric}",
            ]

            # Define color schemes for each scenario
            color_schemes = {
                "Initial": sns.light_palette("#C0C0C0", as_cmap=True),  # Silver/Gray
                "Bounded": sns.light_palette("#6495ED", as_cmap=True),  # Blue
                "Bounded x3": sns.light_palette("#efb400", as_cmap=True),  # Gold
            }

            for i, scenario in enumerate(scenarios):
                data = self._get_allocation_data(scenario)

                # Format data for heatmap with consistent formatting
                plot_data = data.copy()
                plot_data["abs.mean\nspend"] = plot_data["abs.mean\nspend"].apply(
                    lambda x: f"{x/1000:.0f}K"
                )
                for col in metrics[1:]:
                    if col.endswith("%"):
                        plot_data[col] = plot_data[col].apply(lambda x: f"{x*100:.1f}%")
                    elif col == f"mean\nCPA":
                        # Format CPA as a round number
                        plot_data[col] = plot_data[col].apply(
                            lambda x: f"{int(round(x))}"
                        )
                    else:
                        plot_data[col] = plot_data[col].apply(lambda x: f"{x:.2f}")

                # Normalize each column independently for proper color gradients
                norm_data = data.copy()
                for col in metrics:
                    col_values = norm_data[col].values
                    if col_values.any():
                        min_val = col_values.min()
                        max_val = col_values.max()
                        if max_val > min_val:
                            norm_data[col] = (col_values - min_val) / (
                                max_val - min_val
                            )
                        else:
                            norm_data[col] = 0

                # Create heatmap
                sns.heatmap(
                    norm_data[metrics],
                    ax=axes[i],
                    cmap=color_schemes[scenario],
                    annot=plot_data[metrics].values,
                    fmt="",
                    cbar=False,
                    annot_kws={
                        "fontsize": 7,
                        "va": "center",
                    },  # Smaller font, vertically centered
                    linewidths=0.5,  # Add cell borders for better readability
                )

                # Set title above each heatmap
                axes[i].set_title(scenario, fontsize=9, pad=5)

                # Only show y-axis label and tick labels for the first subplot
                if i == 0:
                    axes[i].set_ylabel("Paid Media", fontsize=9)
                else:
                    axes[i].set_ylabel("")
                    axes[i].set_yticks([])  # Hide y ticks for non-first plots

                # Customize x-tick labels
                axes[i].set_xticklabels(metrics, rotation=45, ha="right", fontsize=7)

                # Add thick border between heatmaps (except for the last one)
                if i < len(scenarios) - 1:
                    axes[i].spines["right"].set_visible(True)
                    axes[i].spines["right"].set_color("black")
                    axes[i].spines["right"].set_linewidth(2)

            # Add main title
            plt.suptitle(
                f"Budget Allocation per Paid Media Variable per {self.interval_type}",
                fontsize=10,
                y=0.98,
            )

            # Adjust layout to make room for the title while maintaining zero spacing
            plt.subplots_adjust(top=0.85, wspace=0)

            return fig

        except Exception as e:
            logger.error("Failed to create allocation matrix plot: %s", str(e))
            raise

    def _get_allocation_data(self, scenario: str) -> pd.DataFrame:
        """Prepare data for allocation matrix plot."""
        try:
            if scenario == "Initial":
                # Calculate mean metric differently based on the metric type
                if self.metric == "ROAS":
                    mean_metric = self.dt_optimOut.init_response_unit / np.where(
                        self.dt_optimOut.init_spend_unit > 0,
                        self.dt_optimOut.init_spend_unit,
                        np.inf,
                    )
                else:  # CPA
                    mean_metric = np.where(
                        self.dt_optimOut.init_response_unit > 0,
                        self.dt_optimOut.init_spend_unit
                        / self.dt_optimOut.init_response_unit,
                        np.inf,
                    )

                data = {
                    "abs.mean\nspend": self.dt_optimOut.init_spend_unit,
                    "mean\nspend%": self.dt_optimOut.init_spend_unit
                    / self.dt_optimOut.init_spend_unit.sum(),
                    "mean\nresponse%": self.dt_optimOut.init_response_unit
                    / self.dt_optimOut.init_response_unit.sum(),
                    f"mean\n{self.metric}": mean_metric,
                    # Removed mCPA column
                }

            elif scenario == "Bounded":
                # Calculate mean metric differently based on the metric type
                if self.metric == "ROAS":
                    mean_metric = self.dt_optimOut.optm_response_unit / np.where(
                        self.dt_optimOut.optm_spend_unit > 0,
                        self.dt_optimOut.optm_spend_unit,
                        np.inf,
                    )
                else:  # CPA
                    mean_metric = np.where(
                        self.dt_optimOut.optm_response_unit > 0,
                        self.dt_optimOut.optm_spend_unit
                        / self.dt_optimOut.optm_response_unit,
                        np.inf,
                    )

                data = {
                    "abs.mean\nspend": self.dt_optimOut.optm_spend_unit,
                    "mean\nspend%": self.dt_optimOut.optm_spend_unit
                    / self.dt_optimOut.optm_spend_unit.sum(),
                    "mean\nresponse%": self.dt_optimOut.optm_response_unit
                    / self.dt_optimOut.optm_response_unit.sum(),
                    f"mean\n{self.metric}": mean_metric,
                    # Removed mCPA column
                }

            else:  # "Bounded x3"
                # Calculate mean metric differently based on the metric type
                if self.metric == "ROAS":
                    mean_metric = (
                        self.dt_optimOut.optm_response_unit_unbound
                        / np.where(
                            self.dt_optimOut.optm_spend_unit_unbound > 0,
                            self.dt_optimOut.optm_spend_unit_unbound,
                            np.inf,
                        )
                    )
                else:  # CPA
                    mean_metric = np.where(
                        self.dt_optimOut.optm_response_unit_unbound > 0,
                        self.dt_optimOut.optm_spend_unit_unbound
                        / self.dt_optimOut.optm_response_unit_unbound,
                        np.inf,
                    )

                data = {
                    "abs.mean\nspend": self.dt_optimOut.optm_spend_unit_unbound,
                    "mean\nspend%": self.dt_optimOut.optm_spend_unit_unbound
                    / self.dt_optimOut.optm_spend_unit_unbound.sum(),
                    "mean\nresponse%": self.dt_optimOut.optm_response_unit_unbound
                    / self.dt_optimOut.optm_response_unit_unbound.sum(),
                    f"mean\n{self.metric}": mean_metric,
                    # Removed mCPA column
                }

            return pd.DataFrame(data, index=self.dt_optimOut.channels)

        except Exception as e:
            logger.error(
                "Failed to get allocation data for scenario %s: %s", scenario, str(e)
            )
            raise

    def _plot_response_curves(self) -> plt.Figure:
        """Plot response curves with optimization points."""
        if any(
            attr is None
            for attr in [self.dt_optimOut, self.mainPoints, self.budget_allocator]
        ):
            raise ValueError("All data attributes must be provided")

        logger.info("Creating response curves plot")
        try:
            n_channels = len(self.dt_optimOut.channels)
            n_rows = (n_channels + 2) // 3  # 3 columns
            fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows), squeeze=False)
            axes = axes.flatten()

            scenarios = ["Initial", "Bounded", "Bounded x3"]
            scenario_colors = ["gray", "#4682B4", "#DAA520"]

            # Plot for each channel
            for i, channel in enumerate(self.dt_optimOut.channels):
                ax = axes[i]

                # Get channel parameters
                carryover = (
                    self.budget_allocator.allocator_data_preparer.hill_params.carryover[
                        i
                    ]
                )

                # Generate response curve
                max_spend = np.max(self.mainPoints.spend_points[:, i]) * 1.5
                spend_range = np.linspace(0, max_spend, 100)
                response = self._calculate_response_curve_hill(spend_range, i)

                # Plot grey area for historical carryover
                carryover_mask = spend_range <= carryover
                if any(carryover_mask):
                    ax.fill_between(
                        spend_range[carryover_mask],
                        response[carryover_mask],
                        color="grey",
                        alpha=0.4,
                        label="Historical Carryover",
                    )

                # Plot main curve
                ax.plot(
                    spend_range, response, label=channel, color="#1f77b4", alpha=0.7
                )

                # Add constraint lines
                lower_bound = self.dt_optimOut.init_spend_unit[i] * 0.7
                upper_bound = self.dt_optimOut.init_spend_unit[i] * 1.2

                # Add constraint lines for bounded scenario
                ax.axvline(x=lower_bound, color="gray", linestyle="--", alpha=0.3)
                ax.axvline(x=upper_bound, color="gray", linestyle="--", alpha=0.3)

                # Add extended constraint lines for unbounded scenario
                lower_bound_ext = (
                    self.dt_optimOut.init_spend_unit[i] * 0.1
                )  # Using 0.1 as in R
                upper_bound_ext = (
                    self.dt_optimOut.init_spend_unit[i] * 1.6
                )  # Using 1.6 as in R
                ax.axvline(x=lower_bound_ext, color="gray", linestyle=":", alpha=0.2)
                ax.axvline(x=upper_bound_ext, color="gray", linestyle=":", alpha=0.2)

                # Plot points for each scenario
                for scenario_idx in range(len(scenarios)):
                    ax.scatter(
                        self.mainPoints.spend_points[scenario_idx, i],
                        self.mainPoints.response_points[scenario_idx, i],
                        color=scenario_colors[scenario_idx],
                        marker="o",
                        s=100,
                        label=scenarios[scenario_idx],
                        zorder=5,
                    )

                    # Add spend amount annotation
                    ax.annotate(
                        f"{self.mainPoints.spend_points[scenario_idx, i]:,.0f}",
                        (
                            self.mainPoints.spend_points[scenario_idx, i],
                            self.mainPoints.response_points[scenario_idx, i],
                        ),
                        xytext=(10, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )

                # Customize subplot
                ax.set_title(channel, fontsize=10)
                ax.set_xlabel("Spend")
                ax.set_ylabel("Response")

                # Format axis labels
                ax.yaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: format(int(x), ","))
                )
                ax.xaxis.set_major_formatter(
                    plt.FuncFormatter(lambda x, p: format(int(x), ","))
                )

                # Add legend to first plot only
                if i == 0:
                    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            # Remove empty subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            # Main title
            plt.suptitle(
                f"Simulated Response Curve per {self.interval_type}\n"
                f"Spend per {self.interval_type} (grey area: mean historical carryover)",
                y=1.02,
            )

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error("Failed to create response curves plot: %s", str(e))
            raise

    def _plot_combined_response_curves(
        self, max_projection_spend: float = 50000000
    ) -> plt.Figure:
        """
        Plot response curves for all channels on a single plot for comparison.

        Args:
            max_projection_spend: Maximum spend to project for any channel (default: 50,000,000)
        """
        if any(
            attr is None
            for attr in [self.dt_optimOut, self.mainPoints, self.budget_allocator]
        ):
            raise ValueError("All data attributes must be provided")

        logger.info("Creating combined response curves plot")
        try:
            # Create a single figure and axis
            fig, ax = plt.subplots(figsize=(12, 8))

            # Import Line2D for custom legend entries
            from matplotlib.lines import Line2D

            # Set up colors for channels and scenarios
            channel_colors = plt.cm.tab10(
                np.linspace(0, 1, len(self.dt_optimOut.channels))
            )
            scenario_markers = ["o", "s", "^"]  # Different marker shapes for scenarios
            scenarios = ["Initial", "Bounded", "Bounded x3"]
            scenario_colors = ["gray", "#4682B4", "#DAA520"]  # Match existing colors

            # Plot each channel
            for i, channel in enumerate(self.dt_optimOut.channels):
                # Get initial spend for this channel
                initial_spend = self.dt_optimOut.init_spend_unit[i]

                # Calculate projection range up to 3x initial spend or max_projection_spend
                projection_limit = min(initial_spend * 3, max_projection_spend)

                # Generate historical curve (up to initial spend)
                historical_spend_range = np.linspace(0, initial_spend, 50)
                historical_response = self._calculate_response_curve_hill(
                    historical_spend_range, i
                )

                # Generate projection curve (from initial spend to projection limit)
                projection_spend_range = np.linspace(
                    initial_spend, projection_limit, 50
                )
                projection_response = self._calculate_response_curve_hill(
                    projection_spend_range, i
                )

                # Plot historical curve (solid line)
                ax.plot(
                    historical_spend_range,
                    historical_response,
                    label=channel,
                    color=channel_colors[i],
                    alpha=0.7,
                )

                # Plot projection curve (dotted line)
                ax.plot(
                    projection_spend_range,
                    projection_response,
                    color=channel_colors[i],
                    linestyle=":",
                    alpha=0.7,
                )

                # Plot points only for "Bounded" and "Bounded x3" scenarios (skip "Initial")
                for scenario_idx, scenario in enumerate(
                    scenarios[1:], 1
                ):  # Start from 1 to skip "Initial"
                    ax.scatter(
                        self.mainPoints.spend_points[scenario_idx, i],
                        self.mainPoints.response_points[scenario_idx, i],
                        color=channel_colors[i],
                        marker=scenario_markers[scenario_idx],
                        s=80,
                        alpha=0.8,
                        edgecolors=scenario_colors[scenario_idx],
                        linewidths=1.5,
                    )

            # Create separate legend entries for channels and scenarios
            channel_handles = []
            channel_labels = []
            for i, channel in enumerate(self.dt_optimOut.channels):
                # Create a handle for each channel
                handle = Line2D([0], [0], color=channel_colors[i], lw=1.5)
                channel_handles.append(handle)
                channel_labels.append(channel)

            # Create separate legend entries for scenarios (skip "Initial")
            scenario_handles = []
            scenario_labels = []
            for scenario_idx, scenario in enumerate(scenarios[1:], 1):  # Skip "Initial"
                # Create a handle for each scenario
                handle = Line2D(
                    [0],
                    [0],
                    marker=scenario_markers[scenario_idx],
                    color="none",  # No line
                    markerfacecolor="gray",  # Generic color (not channel-specific)
                    markeredgecolor=scenario_colors[
                        scenario_idx
                    ],  # Edge color matches the scenario
                    markersize=8,
                )
                scenario_handles.append(handle)
                scenario_labels.append(scenario)

            # Create a custom legend entry for projection lines
            projection_handle = Line2D([0], [0], color="black", lw=1.5, linestyle=":")
            projection_label = "Projection (up to 3x initial spend)"

            # Add the channel legend
            leg1 = ax.legend(
                channel_handles,
                channel_labels,
                loc="upper left",
                title="Channels",
                frameon=True,
                fontsize=9,
            )
            ax.add_artist(leg1)

            # Add the scenario legend
            leg2 = ax.legend(
                scenario_handles,
                scenario_labels,
                loc="upper right",
                title="Scenarios",
                frameon=True,
                fontsize=9,
            )
            ax.add_artist(leg2)

            # Add the projection line legend
            leg3 = ax.legend(
                [projection_handle],
                [projection_label],
                loc="lower right",
                frameon=True,
                fontsize=9,
            )
            ax.add_artist(leg3)

            # Customize plot
            ax.set_title(
                f"Comparative Response Curves Across All Channels\n"
                f"Solid: Historical Data | Dotted: Projections (up to 3x initial spend)\n"
                f"Spend per {self.interval_type}"
            )
            ax.set_xlabel("Spend")
            ax.set_ylabel("Response")

            # Format axis labels
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: format(int(x), ","))
            )
            ax.xaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: format(int(x), ","))
            )

            # Add grid for readability
            ax.grid(True, linestyle="--", alpha=0.3)

            plt.tight_layout()
            return fig

        except Exception as e:
            logger.error("Failed to create combined response curves plot: %s", str(e))
            raise

    def _calculate_response_curve_hill(
        self, spend_range: np.ndarray, channel_idx: int
    ) -> np.ndarray:
        """Calculate response values using Hill transformation."""
        try:
            # Get parameters from budget allocator
            alpha = self.budget_allocator.allocator_data_preparer.hill_params.alphas[
                channel_idx
            ]
            coef = self.budget_allocator.allocator_data_preparer.hill_params.coefs[
                channel_idx
            ]
            carryover = (
                self.budget_allocator.allocator_data_preparer.hill_params.carryover[
                    channel_idx
                ]
            )
            gamma = self.budget_allocator.allocator_data_preparer.hill_params.gammas[
                channel_idx
            ]

            # Get channel name
            channel = self.budget_allocator.allocator_data_preparer.media_spend_sorted[
                channel_idx
            ]

            # Calculate inflexion point directly as in data_preparation.py
            adstocked_data = self.budget_allocator.allocator_data_preparer.pareto_result.media_vec_collect[
                self.budget_allocator.allocator_data_preparer.pareto_result.media_vec_collect[
                    "type"
                ]
                == "adstockedMedia"
            ]
            # Get max value for this channel's adstocked data
            max_value = adstocked_data[channel].max()
            inflexion = max_value * gamma

            # Step 1: Adstock transformation
            x_adstocked = spend_range + carryover

            # Step 2: Hill transformation
            x_hill = np.power(x_adstocked, alpha)
            gamma_hill = np.power(inflexion, alpha)

            # Step 3: Response calculation with saturation
            response = coef * (x_hill / (x_hill + gamma_hill))

            return response

        except Exception as e:
            logger.error(
                "Failed to calculate response curve for channel %d: %s",
                channel_idx,
                str(e),
            )
            raise

    def plot_all(
        self, display_plots: bool = True, export_location: Union[str, Path] = None
    ) -> Dict[str, plt.Figure]:
        """
        Create all allocator plots.
        """
        logger.info(f"Creating all plots for model {self.model_id}")

        try:
            plots = {
                "budget_opt": self._plot_budget_comparison(),
                "allocation": self._plot_allocation_matrix(),
                "response": self._plot_response_curves(),
                "combined_response": self._plot_combined_response_curves(
                    max_projection_spend=50000000
                ),  # Add the new plot with parameter
            }

            if display_plots:
                self.display_plots(plots)

            if export_location is not None:
                self.export_plots_fig(export_location, plots)

            return plots

        except Exception as e:
            logger.error("Failed to generate all plots: %s", str(e))
            raise

    def _plot_budget_comparison(self) -> plt.Figure:
        """Plot budget comparison with raw values and summary stats, matching R version."""
        logger.info("Creating budget comparison plot")
        try:
            # Create figure with grid
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.grid(True, axis="y", linestyle="--", alpha=0.7, zorder=0)

            # Get number of periods
            n_periods = int(
                self.dt_optimOut.periods.split()[0]
            )  # e.g., "156 weeks" -> 156

            # Calculate scenarios with total values (not unit values)
            self.scenarios = {
                "Initial": {
                    "spend": np.sum(self.dt_optimOut.init_spend_unit) * n_periods,
                    "response": np.sum(self.dt_optimOut.init_response_unit) * n_periods,
                },
                "Bounded": {
                    "spend": np.sum(self.dt_optimOut.optm_spend_unit) * n_periods,
                    "response": np.sum(self.dt_optimOut.optm_response_unit) * n_periods,
                },
                "Bounded x3": {
                    "spend": np.sum(self.dt_optimOut.optm_spend_unit_unbound)
                    * n_periods,
                    "response": np.sum(self.dt_optimOut.optm_response_unit_unbound)
                    * n_periods,
                },
            }

            # Print debugging information
            logger.debug("Budget comparison values:")
            for scenario, values in self.scenarios.items():
                logger.debug(
                    f"{scenario}: Spend = {values['spend']:,.2f}, Response = {values['response']:,.2f}"
                )

            # Calculate percentage changes and metrics
            for scenario in self.scenarios:
                if scenario != "Initial":
                    self.scenarios[scenario].update(
                        {
                            "spend_pct_change": (
                                self.scenarios[scenario]["spend"]
                                / self.scenarios["Initial"]["spend"]
                                - 1
                            )
                            * 100,
                            "response_pct_change": (
                                self.scenarios[scenario]["response"]
                                / self.scenarios["Initial"]["response"]
                                - 1
                            )
                            * 100,
                        }
                    )
                    if self.metric == "ROAS":
                        self.scenarios[scenario]["metric_value"] = (
                            self.scenarios[scenario]["response"]
                            / self.scenarios[scenario]["spend"]
                        )
                    else:  # CPA
                        self.scenarios[scenario]["metric_value"] = (
                            self.scenarios[scenario]["spend"]
                            / self.scenarios[scenario]["response"]
                        )

            # Define colors for each scenario
            scenario_colors = {
                "Initial": "#C0C0C0",  # Silver
                "Bounded": "#4682B4",  # Steel Blue
                "Bounded x3": "#DAA520",  # Golden Rod
            }

            # Define spacing parameters
            group_width = 1.2
            group_spacing = 2.0
            bar_spacing = 0.4
            bar_width = 0.3

            x_base = np.arange(
                0,
                len(self.scenarios) * (group_spacing + group_width),
                group_spacing + group_width,
            )

            # Plot bars for each scenario
            for i, (scenario, data) in enumerate(self.scenarios.items()):
                x_group = x_base[i]
                x_spend = x_group + (group_width - bar_spacing) / 2
                x_response = x_group + (group_width + bar_spacing) / 2

                spend_bar = ax.bar(
                    x_spend,
                    data["spend"],
                    bar_width,
                    label=scenario,
                    color=scenario_colors[scenario],
                    zorder=3,
                )
                response_bar = ax.bar(
                    x_response,
                    data["response"],
                    bar_width,
                    color=scenario_colors[scenario],
                    zorder=3,
                )

                # Add value labels
                for bar in [spend_bar, response_bar]:
                    height = bar[0].get_height()
                    ax.text(
                        bar[0].get_x() + bar[0].get_width() / 2.0,
                        height,
                        f"{height:,.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        zorder=4,
                    )

                # Add "total spend" and "total response" labels
                for x_pos, label in [
                    (x_spend, "total spend"),
                    (x_response, "total response"),
                ]:
                    ax.text(
                        x_pos - 0.25,
                        ax.get_ylim()[0] * 0.02,
                        label,
                        ha="center",
                        va="top",
                        rotation=45,
                        fontsize=8,
                        zorder=4,
                    )

                # Add metrics text
                if scenario == "Initial":
                    metrics_text = (
                        f"Spend\nResp\n{self.metric}: "
                        f"{data['response']/data['spend']:.2f}"
                    )
                else:
                    metrics_text = (
                        f"Spend: {data['spend_pct_change']:+.1f}%\n"
                        f"Resp: {data['response_pct_change']:+.1f}%\n"
                        f"{self.metric}: {data['metric_value']:.2f}"
                    )

                ax.text(
                    x_group + group_width / 2,
                    ax.get_ylim()[1],
                    metrics_text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                    zorder=4,
                )

            # Customize plot
            ax.set_title(
                f"Total Budget Optimization Result (scaled up to {self.dt_optimOut.periods})",
                pad=20,
                fontsize=10,
                loc="left",
            )

            # Remove axis ticks and labels
            ax.set_xticks([])
            ax.set_ylabel("")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda x, p: format(int(x), ","))
            )
            ax.tick_params(axis="y", labelsize=8)
            ax.set_xticklabels(ax.get_xticks(), rotation=45, ha="right", fontsize=8)

            # Customize legend
            ax.legend(
                title=None,
                loc="upper right",
                bbox_to_anchor=(1, 1.1),
                ncol=3,
                frameon=False,
                fontsize=8,
            )

            # Remove spines
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)

            plt.tight_layout(pad=2.0)
            return fig

        except Exception as e:
            logger.error("Failed to create budget comparison plot: %s", str(e))
            raise
