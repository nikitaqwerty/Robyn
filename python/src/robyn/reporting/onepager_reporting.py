import os
from typing import Dict, Optional, List, Tuple, Union
import warnings
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from robyn.data.entities.holidays_data import HolidaysData
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import gc

from robyn.modeling.entities.pareto_result import ParetoResult
from robyn.modeling.entities.clustering_results import ClusteredResult
from robyn.data.entities.hyperparameters import Hyperparameters
from robyn.data.entities.mmmdata import MMMData
from robyn.data.entities.enums import PlotType

from robyn.visualization.pareto_visualizer import ParetoVisualizer
from robyn.visualization.cluster_visualizer import ClusterVisualizer
from robyn.visualization.response_visualizer import ResponseVisualizer
from robyn.visualization.transformation_visualizer import TransformationVisualizer

logger = logging.getLogger(__name__)


class OnePager:
    def __init__(
        self,
        pareto_result: ParetoResult,
        clustered_result: Optional[ClusteredResult] = None,
        hyperparameter: Optional[Hyperparameters] = None,
        mmm_data: Optional[MMMData] = None,
        holidays_data: Optional[HolidaysData] = None,
    ):
        self.pareto_result = pareto_result
        self.clustered_result = clustered_result
        self.hyperparameter = hyperparameter
        self.mmm_data = mmm_data
        self.holidays_data = holidays_data

        # Default plots using PlotType enum directly
        self.default_plots = [
            PlotType.WATERFALL,
            PlotType.FITTED_VS_ACTUAL,
            PlotType.SPEND_EFFECT,
            PlotType.BOOTSTRAP,
            PlotType.ADSTOCK,
            PlotType.IMMEDIATE_CARRYOVER,
            PlotType.RESPONSE_CURVES,
            PlotType.DIAGNOSTIC,
        ]

    def _setup_plotting_style(self, reduced_quality=True):
        """Configure the plotting style for the one-pager with memory optimization."""
        plt.style.use("default")
        sns.set_theme(style="whitegrid", context="paper")

        # Use more aggressive memory-efficient settings
        dpi_value = 60 if reduced_quality else 100  # Reduced from 72
        save_dpi = 100 if reduced_quality else 300  # Reduced from 150

        plt.rcParams.update(
            {
                "figure.figsize": (16, 20),  # Further reduced size
                "figure.dpi": dpi_value,
                "savefig.dpi": save_dpi,
                "font.size": 10,  # Smaller fonts
                "axes.titlesize": 14,  # Reduced from 16
                "axes.labelsize": 9,  # Reduced from 10
                "xtick.labelsize": 8,  # Reduced from 9
                "ytick.labelsize": 8,  # Reduced from 9
                "legend.fontsize": 8,  # Reduced from 9
                "figure.titlesize": 12,  # Reduced from 14
                "axes.grid": True,
                "grid.alpha": 0.3,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "savefig.bbox": "tight",  # Use tight bbox by default
                "savefig.pad_inches": 0.3,  # Reduced padding
            }
        )

    def _safe_format(self, value, precision: int = 4) -> str:
        """Safely format numeric values with specified precision."""
        try:
            if isinstance(value, (pd.DataFrame, pd.Series)):
                value = (
                    value.iloc[0] if isinstance(value, pd.Series) else value.iloc[0, 0]
                )
            if pd.isna(value):
                return "0.0000"
            return f"{float(value):.{precision}f}"
        except (TypeError, ValueError, IndexError):
            return "0.0000"

    def _get_model_info(self, solution_id: str) -> Dict[str, str]:
        """Get model performance metrics for specific solution."""
        try:
            x_decomp_agg = self.pareto_result.x_decomp_agg
            plot_media_share = x_decomp_agg[
                (x_decomp_agg["sol_id"] == solution_id)
                & (
                    x_decomp_agg["rn"].isin(
                        self.mmm_data.mmmdata_spec.paid_media_spends
                    )
                )
            ]

            if plot_media_share.empty:
                logger.warning(f"No media share data found for solution {solution_id}")
                return {
                    "rsq_train": "0.0000",
                    "nrmse_train": "0.0000",
                    "decomp_rssd": "0.0000",
                    "formatted_text": "No metrics available",
                }

            metrics = {}

            # Get training metrics
            metrics["rsq_train"] = self._safe_format(
                plot_media_share["rsq_train"].iloc[0]
            )
            metrics["nrmse_train"] = self._safe_format(
                plot_media_share["nrmse_train"].iloc[0]
            )

            # Get validation metrics if available
            if "rsq_val" in plot_media_share.columns:
                metrics["rsq_val"] = self._safe_format(
                    plot_media_share["rsq_val"].iloc[0]
                )
                metrics["nrmse_val"] = self._safe_format(
                    plot_media_share["nrmse_val"].iloc[0]
                )

            # Get test metrics if available
            if "rsq_test" in plot_media_share.columns:
                metrics["rsq_test"] = self._safe_format(
                    plot_media_share["rsq_test"].iloc[0]
                )
                metrics["nrmse_test"] = self._safe_format(
                    plot_media_share["nrmse_test"].iloc[0]
                )

            # Get decomp.rssd
            metrics["decomp_rssd"] = self._safe_format(
                plot_media_share["decomp.rssd"].iloc[0]
            )

            # Get MAPE if available
            if "mape" in plot_media_share.columns:
                metrics["mape"] = self._safe_format(plot_media_share["mape"].iloc[0])

            # Calculate performance (ROAS/CPA)
            dep_var_type = self.mmm_data.mmmdata_spec.dep_var_type
            type_metric = "CPA" if dep_var_type == "conversion" else "ROAS"

            # Use copy to avoid SettingWithCopyWarning
            temp_data = x_decomp_agg[
                (x_decomp_agg["sol_id"] == solution_id)
                & (
                    x_decomp_agg["rn"].isin(
                        self.mmm_data.mmmdata_spec.paid_media_spends
                    )
                )
            ].copy()

            if not temp_data.empty:
                perf = temp_data.groupby("sol_id").agg(
                    {"xDecompAgg": "sum", "total_spend": "sum"}
                )

                if not perf.empty:
                    if type_metric == "ROAS":
                        performance = (
                            perf["xDecompAgg"].iloc[0] / perf["total_spend"].iloc[0]
                        )
                    else:  # CPA
                        performance = (
                            perf["total_spend"].iloc[0] / perf["xDecompAgg"].iloc[0]
                        )
                    metrics["performance"] = f"{performance:.3g} {type_metric}"

            # Format metrics text
            if "rsq_val" in metrics:
                metrics_text = (
                    f"Adj.R2: train = {metrics['rsq_train']}, "
                    f"val = {metrics['rsq_val']}, "
                    f"test = {metrics['rsq_test']} | "
                    f"NRMSE: train = {metrics['nrmse_train']}, "
                    f"val = {metrics['nrmse_val']}, "
                    f"test = {metrics['nrmse_test']} | "
                    f"DECOMP.RSSD = {metrics['decomp_rssd']}"
                )
            else:
                metrics_text = (
                    f"Adj.R2: train = {metrics['rsq_train']} | "
                    f"NRMSE: train = {metrics['nrmse_train']} | "
                    f"DECOMP.RSSD = {metrics['decomp_rssd']}"
                )

            if "mape" in metrics:
                metrics_text += f" | MAPE = {metrics['mape']}"

            if "performance" in metrics:
                metrics_text += f" | {metrics['performance']}"

            metrics["formatted_text"] = metrics_text

            return metrics

        except Exception as e:
            logger.error(
                f"Error getting model info for solution {solution_id}: {str(e)}"
            )
            return {
                "rsq_train": "0.0000",
                "nrmse_train": "0.0000",
                "decomp_rssd": "0.0000",
                "formatted_text": "Error calculating metrics",
            }

    def _add_title_and_metrics(self, fig: plt.Figure, solution_id: str) -> None:
        """Add title and metrics text to the figure."""
        try:
            model_info = self._get_model_info(solution_id)
            metrics_text = model_info.get("formatted_text", "")

            # Add title with larger font and bold
            fig.suptitle(
                f"MMM Analysis One-Pager for Model: {solution_id}",
                fontsize=20,  # Reduced from 24
                y=0.98,
                weight="bold",
                fontfamily="sans-serif",
            )

            # Add metrics text if available
            if metrics_text:
                fig.text(
                    0.5,  # Center horizontally
                    0.955,  # Position below title
                    metrics_text,
                    fontsize=12,  # Reduced from 16
                    ha="center",
                    va="top",
                    weight="bold",
                )
        except Exception as e:
            logger.error(f"Error adding title and metrics: {str(e)}")
            # Fallback title with same style
            fig.suptitle(
                f"MMM Analysis One-Pager for Model: {solution_id}",
                fontsize=20,  # Reduced from 24
                y=0.98,
                weight="bold",
                fontfamily="sans-serif",
            )

    def _generate_solution_plots(
        self, solution_id: str, plots: List[PlotType], gs: GridSpec
    ) -> None:
        """Generate plots for a single solution with dynamic layout."""
        try:
            # Validate plot types
            for plot in plots:
                if not isinstance(plot, PlotType):
                    logger.error(
                        f"Invalid plot type provided: {plot}. Must be PlotType enum."
                    )
                    raise TypeError(
                        f"Plot type must be PlotType enum, got {type(plot)}"
                    )

            # Initialize visualizers
            pareto_viz = None
            cluster_viz = None
            response_viz = None
            transfor_viz = None

            # Define the plots that need each visualizer
            pareto_plots = [
                PlotType.WATERFALL,
                PlotType.FITTED_VS_ACTUAL,
                PlotType.DIAGNOSTIC,
                PlotType.IMMEDIATE_CARRYOVER,
                PlotType.ADSTOCK,
            ]

            # Memory optimization: Only initialize visualizers that are actually needed
            needed_visualizers = set()
            for plot in plots:
                if plot in pareto_plots:
                    needed_visualizers.add("pareto")
                elif plot == PlotType.BOOTSTRAP:
                    needed_visualizers.add("cluster")
                elif plot == PlotType.RESPONSE_CURVES:
                    needed_visualizers.add("response")
                elif plot == PlotType.SPEND_EFFECT:
                    needed_visualizers.add("transform")

            # Initialize only the necessary visualizers
            if (
                "pareto" in needed_visualizers
                and self.hyperparameter
                and self.holidays_data
            ):
                pareto_viz = ParetoVisualizer(
                    self.pareto_result,
                    self.mmm_data,
                    self.holidays_data,
                    self.hyperparameter,
                )

            if "cluster" in needed_visualizers and self.clustered_result:
                cluster_viz = ClusterVisualizer(
                    self.pareto_result, self.clustered_result, self.mmm_data
                )

            if "response" in needed_visualizers:
                response_viz = ResponseVisualizer(self.pareto_result, self.mmm_data)

            if "transform" in needed_visualizers:
                transfor_viz = TransformationVisualizer(
                    self.pareto_result, self.mmm_data
                )

            # Define plot configurations
            plot_config = {
                PlotType.SPEND_EFFECT: {
                    "title": "Share of Total Spend, Effect & Performance",
                    "func": lambda ax: (
                        transfor_viz.generate_spend_effect_comparison(solution_id, ax)
                        if transfor_viz
                        else None
                    ),
                },
                PlotType.WATERFALL: {
                    "title": "Response Decomposition Waterfall",
                    "func": lambda ax: (
                        pareto_viz.generate_waterfall(solution_id, ax)
                        if pareto_viz
                        else None
                    ),
                },
                PlotType.FITTED_VS_ACTUAL: {
                    "title": "Actual vs. Predicted Response",
                    "func": lambda ax: (
                        pareto_viz.generate_fitted_vs_actual(solution_id, ax)
                        if pareto_viz
                        else None
                    ),
                },
                PlotType.DIAGNOSTIC: {
                    "title": "Fitted vs. Residual",
                    "func": lambda ax: (
                        pareto_viz.generate_diagnostic_plot(solution_id, ax)
                        if pareto_viz
                        else None
                    ),
                },
                PlotType.IMMEDIATE_CARRYOVER: {
                    "title": "Immediate vs. Carryover Response Percentage",
                    "func": lambda ax: (
                        pareto_viz.generate_immediate_vs_carryover(solution_id, ax)
                        if pareto_viz
                        else None
                    ),
                },
                PlotType.ADSTOCK: {
                    "title": "Adstock Rate Analysis",
                    "func": lambda ax: (
                        pareto_viz.generate_adstock_rate(solution_id, ax)
                        if pareto_viz
                        else None
                    ),
                },
                PlotType.BOOTSTRAP: {
                    "title": "Bootstrapped Performance Metrics",
                    "func": lambda ax: (
                        cluster_viz.generate_bootstrap_confidence(solution_id, ax)
                        if cluster_viz
                        else None
                    ),
                },
                PlotType.RESPONSE_CURVES: {
                    "title": "Response Curves and Mean Spends by Channel",
                    "func": lambda ax: (
                        response_viz.generate_response_curves(solution_id, ax)
                        if response_viz
                        else None
                    ),
                },
            }

            # Create plots with dynamic positioning - process plots in batches to conserve memory
            for i, plot_type in enumerate(plots):
                if plot_type not in plot_config:
                    logger.error(f"Unsupported plot type: {plot_type}")
                    continue

                row = i // 2
                col = i % 2

                config = plot_config[plot_type]
                try:
                    ax = plt.subplot(gs[row, col])
                    config["func"](ax)
                    ax.set_title(f"{config['title']} (Solution {solution_id})")
                except Exception as e:
                    logger.error(
                        f"Failed to generate plot {plot_type.name} for solution {solution_id}: {str(e)}",
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"Error generating {plot_type.name}",
                        ha="center",
                        va="center",
                    )

                # Force garbage collection after each plot
                plt.figure().clear()
                plt.close()
                gc.collect()

            # Clear visualizers to free memory
            del pareto_viz, cluster_viz, response_viz, transfor_viz
            gc.collect()

        except Exception as e:
            logger.error(
                f"Fatal error generating plots for solution {solution_id}: {str(e)}",
                exc_info=True,
            )
            raise

    def generate_one_pager(
        self,
        solution_ids: Union[str, List[str]] = "all",
        plots: Optional[List[PlotType]] = None,
        figsize: tuple = (16, 20),  # Further reduced default
        save_path: Optional[str] = None,
        top_pareto: bool = False,
        max_solutions: int = 1,  # Reduced from 5 to 1 to prevent memory issues
        reduced_quality: bool = True,  # New parameter to control quality vs memory usage
    ) -> List[plt.Figure]:
        """Generate separate one-pager for each solution ID with memory optimizations.

        Args:
            solution_ids: Single solution ID or list of solution IDs or 'all'
            plots: Optional list of plot types from PlotType enum
            figsize: Figure size for each page
            save_path: Optional path to save the figures
            top_pareto: If True, loads the one-page summaries for the top Pareto models
            max_solutions: Maximum number of solutions to process (default: 1)
            reduced_quality: If True, use lower DPI and image quality to save memory

        Returns:
            List[plt.Figure]: List of generated figures, one per solution
        """
        # Explicitly clean up before starting
        plt.close("all")
        gc.collect()

        # Setup plotting style with quality considerations
        self._setup_plotting_style(reduced_quality=reduced_quality)

        # Normalize solution ID column name at the start
        if "solID" in self.pareto_result.x_decomp_agg.columns:
            self.pareto_result.x_decomp_agg = self.pareto_result.x_decomp_agg.rename(
                columns={"solID": "sol_id"}
            )

        # Convert string plot types to PlotType if necessary
        if plots and isinstance(plots[0], str):
            try:
                plots = [PlotType[plot.upper()] for plot in plots]
            except KeyError as e:
                raise ValueError(f"Invalid plot type: {e}")

        # Use default plots if none provided
        plots = plots or self.default_plots

        # Memory optimization: Limit number of plots if memory is a concern
        if reduced_quality and len(plots) > 10:
            logger.warning("Reducing number of plots to save memory")
            plots = plots[:10]  # Keep only first 10 plots

        # Handle solution IDs based on top_pareto parameter
        if top_pareto:
            if self.clustered_result is None or not hasattr(
                self.clustered_result, "top_solutions"
            ):
                raise ValueError("No clustered results or top solutions available")
            try:
                # Try accessing 'sol_id' column if it's a DataFrame
                if isinstance(self.clustered_result.top_solutions, pd.DataFrame):
                    solution_ids = self.clustered_result.top_solutions[
                        "sol_id"
                    ].tolist()
                elif isinstance(self.clustered_result.top_solutions, pd.Series):
                    solution_ids = self.clustered_result.top_solutions.tolist()
                elif isinstance(self.clustered_result.top_solutions, list):
                    solution_ids = self.clustered_result.top_solutions
                else:
                    raise ValueError(
                        f"Unexpected type for top_solutions: {type(self.clustered_result.top_solutions)}"
                    )
                solution_ids = [
                    str(sid)
                    for sid in solution_ids
                    if sid is not None and pd.notna(sid)
                ]
                if not solution_ids:
                    raise ValueError("No valid solution IDs found in top solutions")
                logger.debug(f"Loading {len(solution_ids)} top solutions")

            except Exception as e:
                raise ValueError(f"Error processing top solutions: {str(e)}")
        else:
            if solution_ids == "all":
                solution_ids = list(self.pareto_result.plot_data_collect.keys())
            elif isinstance(solution_ids, str):
                solution_ids = [solution_ids]
            elif isinstance(solution_ids, (list, tuple)):
                solution_ids = list(solution_ids)
            else:
                raise ValueError(
                    f"solution_ids must be string or list/tuple, got {type(solution_ids)}"
                )

        # Limit number of solutions to prevent memory issues
        if len(solution_ids) > max_solutions:
            warnings.warn(
                f"Too many solutions to process ({len(solution_ids)}). "
                f"Limiting to {max_solutions} solutions to prevent memory issues."
            )
            solution_ids = solution_ids[:max_solutions]

        # Validate solution IDs
        invalid_ids = [
            sid
            for sid in solution_ids
            if sid not in self.pareto_result.plot_data_collect
        ]
        if invalid_ids:
            raise ValueError(f"Invalid solution IDs: {invalid_ids}")

        figures = []
        try:
            if save_path:
                os.makedirs(save_path, exist_ok=True)

            for i, solution_id in enumerate(solution_ids):
                logger.debug(
                    f"Generating one-pager for solution {solution_id} ({i+1}/{len(solution_ids)})"
                )

                try:
                    # Close any existing figures to prevent memory leaks
                    plt.close("all")
                    gc.collect()  # Force garbage collection

                    n_plots = len(plots)
                    n_rows = (n_plots + 1) // 2  # Ceiling division for number of rows

                    # Create figure with the specified size
                    fig = plt.figure(figsize=figsize)

                    # Create GridSpec with explicit spacing parameters
                    gs = GridSpec(
                        n_rows,
                        2,
                        figure=fig,
                        height_ratios=[1] * n_rows,
                        top=0.92,  # Space for title and metrics
                        bottom=0.05,  # Extend plots to bottom
                        left=0.08,  # Left margin
                        right=0.92,  # Right margin
                        hspace=0.4,  # Vertical space between subplots
                        wspace=0.2,  # Horizontal space between subplots
                    )

                    # Generate plots for this solution
                    self._generate_solution_plots(solution_id, plots, gs)

                    # Add title and metrics
                    self._add_title_and_metrics(fig, solution_id)

                    if save_path:
                        save_file = os.path.join(
                            save_path, f"onepager_solution_{solution_id}.png"
                        )
                        # Use lower quality for saving to reduce file size and memory usage
                        dpi = (
                            80 if reduced_quality else 150
                        )  # Even lower dpi for saving
                        fig.savefig(
                            save_file,
                            bbox_inches="tight",
                            pad_inches=0.3,
                            dpi=dpi,
                        )
                        logger.debug(f"Saved figure to {save_file}")

                    # Store the figure only if needed for return
                    if not save_path:
                        figures.append(fig)
                    else:
                        # Close figure after saving to free memory
                        plt.close(fig)
                        # Just store the filename if saved to disk
                        figures.append(save_file)

                    # Force garbage collection after each solution
                    gc.collect()

                except Exception as e:
                    logger.error(
                        f"Error generating plot for solution {solution_id}: {str(e)}",
                        exc_info=True,
                    )
                    plt.close("all")  # Make sure to close any open figures

        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}", exc_info=True)
            # Close all figures to free memory
            plt.close("all")
            raise

        return figures
