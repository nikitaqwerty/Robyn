import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Any, Union
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from robyn.allocator.allocator import BudgetAllocator
from robyn.visualization.base_visualizer import BaseVisualizer

logger = logging.getLogger(__name__)


class AllocatorVisualizer(BaseVisualizer):
    """Generates plots for Robyn allocator results."""

    def __init__(
        self,
        budget_allocator: BudgetAllocator,
        metric: str = "ROAS",
        quiet: bool = False,
    ):
        super().__init__()
        logger.info("Initializing AllocatorPlotter")
        self.dt_optim_out = budget_allocator.dt_optim_out
        self.eval_list = budget_allocator.eval_dict
        self.metric = metric
        self.budget_allocator = budget_allocator
        self.logger = logging.getLogger(__name__)

    def _plot_response_spend_comparison(self) -> go.Figure:
        """Creates the response and spend comparison plot."""
        # Initial values
        init_total_spend = self.dt_optim_out["initSpendTotal"].iloc[0]
        init_total_response = self.dt_optim_out["initResponseTotal"].iloc[0]
        init_total_roi = init_total_response / init_total_spend
        init_total_cpa = init_total_spend / init_total_response
        # Bounded optimization values
        optm_total_spend_bounded = self.dt_optim_out["optmSpendTotal"].iloc[0]
        optm_total_response_bounded = self.dt_optim_out["optmResponseTotal"].iloc[0]
        optm_total_roi_bounded = optm_total_response_bounded / optm_total_spend_bounded
        optm_total_cpa_bounded = optm_total_spend_bounded / optm_total_response_bounded
        # Unbounded optimization values
        optm_total_spend_unbounded = self.dt_optim_out["optmSpendTotalUnbound"].iloc[0]
        optm_total_response_unbounded = self.dt_optim_out[
            "optmResponseTotalUnbound"
        ].iloc[0]
        optm_total_roi_unbounded = (
            optm_total_response_unbounded / optm_total_spend_unbounded
        )
        optm_total_cpa_unbounded = (
            optm_total_spend_unbounded / optm_total_response_unbounded
        )
        bound_mult = self.dt_optim_out["unconstr_mult"].iloc[0]

        # Check if optimization topped out
        optm_topped_bounded = optm_topped_unbounded = any_topped = False
        if self.eval_list.get("total_budget") is not None:
            total_budget = self.eval_list["total_budget"]
            optm_topped_bounded = round(optm_total_spend_bounded) < round(total_budget)
            optm_topped_unbounded = round(optm_total_spend_unbounded) < round(
                total_budget
            )
            any_topped = optm_topped_bounded or optm_topped_unbounded
            if optm_topped_bounded and not self.quiet:
                print(
                    "NOTE: Given the upper/lower constrains, the total budget can't be fully allocated (^)"
                )

        # Get levs1 from eval_list
        levs1 = self.eval_list.get(
            "levs1", ["Initial", "Bounded", f"Bounded x{bound_mult}"]
        )

        # If second and third levels are the same, add a space to third level
        if levs1[1] == levs1[2]:
            levs1[2] = f"{levs1[2]} "

        # Create levs2 based on scenario
        if self.budget_allocator.params.scenario == "max_response":
            levs2 = [
                "Initial",
                f"Bounded{'ˆ' if optm_topped_bounded else ''}",
                f"Bounded{'ˆ' if optm_topped_unbounded else ''} x{bound_mult}",
            ]
        else:  # target_efficiency
            levs2 = levs1

        # Create response metric dataframe
        self.resp_metric = pd.DataFrame(
            {
                "type": pd.Categorical(
                    levs1, categories=levs1
                ),  # Make it a factor with levels
                "type_lab": pd.Categorical(
                    levs2, categories=levs2
                ),  # Make it a factor with levels
                "total_spend": [
                    init_total_spend,
                    optm_total_spend_bounded,
                    optm_total_spend_unbounded,
                ],
                "total_response": [
                    init_total_response,
                    optm_total_response_bounded,
                    optm_total_response_unbounded,
                ],
                "total_response_lift": [
                    0,
                    self.dt_optim_out["optmResponseUnitTotalLift"].iloc[0],
                    self.dt_optim_out["optmResponseUnitTotalLiftUnbound"].iloc[0],
                ],
                "total_roi": [
                    init_total_roi,
                    optm_total_roi_bounded,
                    optm_total_roi_unbounded,
                ],
                "total_cpa": [
                    init_total_cpa,
                    optm_total_cpa_bounded,
                    optm_total_cpa_unbounded,
                ],
            }
        )

        # Create df_roi (similar to R's df_roi transformation)
        df_spend = self.resp_metric[["type", "total_spend"]].rename(
            columns={"total_spend": "value"}
        )
        df_spend["name"] = "total spend"
        df_response = self.resp_metric[["type", "total_response"]].rename(
            columns={"total_response": "value"}
        )
        df_response["name"] = "total response"
        df_roi = pd.concat([df_spend, df_response])

        # Calculate normalized values (matching R's logic)
        df_roi["value_norm"] = df_roi.apply(
            lambda x: (
                x["value"]
                if self.metric == "ROAS"
                else x["value"] / df_roi[df_roi["name"] == x["name"]].iloc[0]["value"]
            ),
            axis=1,
        )

        # Create subplot titles (matching R's labs)
        subplot_titles = [
            f"Initial<br>"
            f"Spend: {self._format_num(0)}<br>"
            f"Resp: {self._format_num(0)}<br>"
            f"{self.metric}: {round(init_total_cpa if self.metric == 'CPA' else init_total_roi, 2)}",
            f"Bounded{'^' if optm_topped_bounded else ''}<br>"
            f"Spend: {self._format_num(100 * (optm_total_spend_bounded - init_total_spend) / init_total_spend)}<br>"
            f"Resp: {self._format_num(100 * self.resp_metric['total_response_lift'].iloc[1])}<br>"
            f"{self.metric}: {round(optm_total_cpa_bounded if self.metric == 'CPA' else optm_total_roi_bounded, 2)}",
            f"Bounded x{bound_mult}{'^' if optm_topped_unbounded else ''}<br>"
            f"Spend: {self._format_num(100 * (optm_total_spend_unbounded - init_total_spend) / init_total_spend)}<br>"
            f"Resp: {self._format_num(100 * self.resp_metric['total_response_lift'].iloc[2])}<br>"
            f"{self.metric}: {round(optm_total_cpa_unbounded if self.metric == 'CPA' else optm_total_roi_unbounded, 2)}",
        ]

        # Create plot with improved layout
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.05,  # Reduce spacing between all columns
        )

        # Define colors matching the reference
        colors = {
            "Initial": "#C0C0C0",  # Silver
            "Bounded": "#4682B4",  # Steel Blue
            f"Bounded x{bound_mult}": "#DAA520",  # Golden Rod
        }

        # Define spacing parameters (adapted for plotly)
        bar_width = 0.2  # Make bars narrower

        # Plot bars for each type
        for i, type_val in enumerate(self.resp_metric["type"]):
            type_data = df_roi[df_roi["type"] == type_val]

            # Calculate x positions for spend and response bars
            x_positions = ["total spend", "total response"]

            fig.add_trace(
                go.Bar(
                    x=type_data["name"],
                    y=type_data["value_norm"],
                    name=type_val,
                    marker_color=colors[type_val],
                    text=[
                        self._format_num(val, abbr=True) for val in type_data["value"]
                    ],
                    textposition="outside",
                    textfont=dict(size=8),
                    showlegend=True,  # Show legend for all groups
                    width=bar_width,
                    hoverinfo="none",
                ),
                row=1,
                col=i + 1,
            )

        # Update layout with improved formatting
        y_max = df_roi["value_norm"].max() * 1.2
        fig.update_layout(
            title={
                "text": f"Total Budget Optimization Result (scaled up to {self.dt_optim_out['periods'].iloc[0]})",
                "y": 0.95,
                "x": 0.02,  # Left-aligned title
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 10},
            },
            showlegend=True,
            legend={
                "orientation": "h",
                "yanchor": "top",  # Keep as top
                "y": 0.9,  # Higher value to position near the top
                "xanchor": "left",
                "x": 0.02,
                "font": {"size": 10},
            },
            height=500,
            width=1000,
            template="plotly_white",
            margin=dict(t=120, b=50, l=50, r=50),
            bargap=0.01,  # Reduce gap between bars within groups (make them closer)
            bargroupgap=0.5,  # Keep the gap between groups
        )

        # Update axes
        for i in range(1, 4):
            # Update y-axes
            fig.update_yaxes(
                range=[0, y_max],
                showticklabels=False,
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.2)",
                showline=False,  # Hide y-axis line
                row=1,
                col=i,
            )

            # Update x-axes
            fig.update_xaxes(
                tickangle=45,
                tickfont=dict(size=8),
                showline=False,  # Hide x-axis line
                row=1,
                col=i,
            )

        # Update subplot titles font size and position
        for annotation in fig["layout"]["annotations"]:
            if annotation["text"] in subplot_titles:
                annotation.update(
                    {
                        "font": dict(size=10, weight="bold"),  # Make text bold
                        "y": 1.08,  # Lower the text position
                        "yanchor": "bottom",
                        "xanchor": "center",
                        "align": "center",
                    }
                )

        return fig

    @staticmethod
    def _format_num(
        num: float,
        signif: int = 3,
        abbr: bool = False,
        pos: str = "%",
        sign: bool = True,
    ) -> str:
        """Format numbers for display."""
        if abbr:
            if abs(num) >= 1e9:
                return f"{num/1e9:.1f}B"
            if abs(num) >= 1e6:
                return f"{num/1e6:.1f}M"
            if abs(num) >= 1e3:
                return f"{num/1e3:.1f}K"
            return f"{num:.1f}"

        formatted = f"{num:.{signif}g}"
        if sign and num > 0:
            formatted = "+" + formatted
        if pos:
            formatted += pos
        return formatted

    def _plot_allocation_comparison(self) -> plt.Figure:
        """Create response and spend comparison plot as a matrix of heatmaps."""

        # Create the base dataframe for plotting
        df_plots = pd.DataFrame()

        # Response share data
        response_share = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initResponseUnitShare"],
                "Bounded": self.dt_optim_out["optmResponseUnitShare"],
                "Unbounded": self.dt_optim_out["optmResponseUnitShareUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="response_share")

        # Spend share data
        spend_share = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initSpendShare"],
                "Bounded": self.dt_optim_out["optmSpendShareUnit"],
                "Unbounded": self.dt_optim_out["optmSpendShareUnitUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="spend_share")

        # Mean spend data
        mean_spend = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initSpendUnit"],
                "Bounded": self.dt_optim_out["optmSpendUnit"],
                "Unbounded": self.dt_optim_out["optmSpendUnitUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="mean_spend")

        # Mean response data - Add this to ensure we have response values for metric calculation
        mean_response = pd.DataFrame(
            {
                "channel": self.dt_optim_out["channels"],
                "Initial": self.dt_optim_out["initResponseUnit"],
                "Bounded": self.dt_optim_out["optmResponseUnit"],
                "Unbounded": self.dt_optim_out["optmResponseUnitUnbound"],
            }
        ).melt(id_vars=["channel"], var_name="type", value_name="mean_response")

        # Combine all dataframes
        df_plots = response_share.merge(spend_share, on=["channel", "type"])
        df_plots = df_plots.merge(mean_spend, on=["channel", "type"])
        df_plots = df_plots.merge(
            mean_response, on=["channel", "type"]
        )  # Add response values

        # Calculate the metric value directly from mean_spend and mean_response
        # This ensures proper calculation of CPA (spend/response) or ROAS (response/spend)
        df_plots["metric_value"] = df_plots.apply(
            lambda x: (
                (x["mean_response"] / x["mean_spend"] if x["mean_spend"] > 0 else 0)
                if self.metric == "ROAS"
                else (
                    x["mean_spend"] / x["mean_response"]
                    if x["mean_response"] > 0
                    else float("inf")
                )
            ),
            axis=1,
        )

        # Handle infinity values for cleaner visualization
        df_plots["metric_value"] = df_plots["metric_value"].replace(
            [np.inf, -np.inf], 1e9
        )

        # Update metrics to match format - 4 columns with proper metric names
        metrics = [
            "abs.mean\nspend",
            "mean\nspend%",
            "mean\nresponse%",
            f"mean\n{self.metric}",
        ]

        # Prepare data for plotting with consistent structure
        plot_data = []
        for metric_name in metrics:
            if metric_name == "abs.mean\nspend":
                values = df_plots["mean_spend"]
            elif metric_name == "mean\nspend%":
                values = df_plots["spend_share"]
            elif metric_name == "mean\nresponse%":
                values = df_plots["response_share"]
            else:  # mean\nROAS or mean\nCPA
                values = df_plots["metric_value"]

            temp_df = pd.DataFrame(
                {
                    "channel": df_plots["channel"],
                    "type": df_plots["type"],
                    "metric": metric_name,
                    "values": values,
                }
            )
            plot_data.append(temp_df)

        df_plot_share = pd.concat(plot_data)

        # Format values
        df_plot_share["values"] = df_plot_share["values"].fillna(0)
        df_plot_share["values"] = df_plot_share["values"].replace([np.inf, -np.inf], 0)
        df_plot_share["values"] = df_plot_share["values"].clip(upper=1e15)

        # Create labels with improved formatting
        df_plot_share["values_label"] = df_plot_share.apply(
            lambda x: (
                f"{x['values']:,.1f}"  # Add comma for thousands
                if x["metric"] == "abs.mean\nspend"
                else (
                    f"{x['values']*100:.1f}%"
                    if x["metric"] in ["mean\nspend%", "mean\nresponse%"]
                    else (
                        f"{int(round(x['values']))}"  # Round CPA to integer
                        if self.metric == "CPA"
                        and x["metric"] == f"mean\n{self.metric}"
                        else f"{x['values']:.2f}"  # 2 decimal places for ROAS
                    )
                )
            ),
            axis=1,
        )

        # Start the plotting with the improved approach
        try:
            # Create figure with GridSpec for precise layout control
            fig = plt.figure(figsize=(15, 6))
            gs = fig.add_gridspec(1, 3, wspace=0)  # Zero spacing between columns
            axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

            # Define color schemes for each scenario
            color_schemes = {
                "Initial": sns.light_palette("#C0C0C0", as_cmap=True),  # Silver/Gray
                "Bounded": sns.light_palette("#6495ED", as_cmap=True),  # Blue
                "Unbounded": sns.light_palette("#efb400", as_cmap=True),  # Gold
            }

            # Plot for each scenario (Initial, Bounded, Unbounded)
            for i, scenario in enumerate(["Initial", "Bounded", "Unbounded"]):
                scenario_data = df_plot_share[df_plot_share["type"] == scenario]

                # Create pivot table
                pivot_data = scenario_data.pivot(
                    index="channel", columns="metric", values="values"
                )
                pivot_data = pivot_data[metrics]  # Reorder columns

                # Create display values
                display_data = scenario_data.pivot(
                    index="channel", columns="metric", values="values_label"
                )
                display_data = display_data[metrics]  # Reorder columns

                # Normalize each column for color intensity
                norm_data = pivot_data.copy()
                for col in metrics:
                    col_values = norm_data[col].values
                    if len(col_values) > 0 and col_values.max() > col_values.min():
                        norm_data[col] = (col_values - col_values.min()) / (
                            col_values.max() - col_values.min()
                        )
                    else:
                        norm_data[col] = 0

                # Create heatmap
                sns.heatmap(
                    norm_data,
                    ax=axes[i],
                    cmap=color_schemes[scenario],
                    annot=display_data.values,
                    fmt="",
                    cbar=False,
                    annot_kws={"fontsize": 7, "va": "center"},
                    linewidths=0.5,  # Add cell borders for better readability
                )

                # Customize axis
                axes[i].set_title(scenario, fontsize=9, pad=5)
                if i == 0:
                    axes[i].set_ylabel("Paid Media", fontsize=9)
                else:
                    axes[i].set_ylabel("")
                    axes[i].set_yticks([])  # Hide y ticks for non-first plots

                axes[i].set_xlabel("")  # Remove the "metric" label

                # Rotate x-axis labels
                axes[i].set_xticklabels(
                    axes[i].get_xticklabels(),
                    rotation=45,
                    horizontalalignment="right",
                    fontsize=7,
                )

                # Add thick border between heatmaps (except for the last one)
                if i < 2:  # 2 is the last index (0, 1, 2)
                    axes[i].spines["right"].set_visible(True)
                    axes[i].spines["right"].set_color("black")
                    axes[i].spines["right"].set_linewidth(2)

            plt.suptitle(
                f"Budget Allocation per Paid Media Variable per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type}",
                fontsize=10,
                y=0.98,
            )

            # Adjust layout to make room for the title while maintaining zero spacing
            plt.subplots_adjust(top=0.85, wspace=0)
            plt.tight_layout(pad=2.0)

            # Add this line to prevent double display
            plt.close(fig)

            return fig

        except Exception as e:
            self.logger.error("Failed to create allocation comparison plot: %s", str(e))
            raise

    def _plot_response_curves(self):
        """Create response curves plot."""
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # 1. Create constraint labels
        constr_labels = []
        for _, row in self.dt_optim_out.iterrows():
            label = (
                f"{row['channels']}\n"
                f"[{row['constr_low']} - {row['constr_up']}] & "
                f"[{round(row['constr_low_unb'], 1)} - {round(row['constr_up_unb'], 1)}]"
            )
            constr_labels.append(
                {
                    "channel": row["channels"],
                    "constr_label": label,
                    "constr_low_abs": row["constr_low_abs"],
                    "constr_up_abs": row["constr_up_abs"],
                    "constr_low_unb_abs": row["constr_low_unb_abs"],
                    "constr_up_unb_abs": row["constr_up_unb_abs"],
                }
            )
        constr_labels = pd.DataFrame(constr_labels)

        # 2. Merge plotDT_scurve with constraint labels
        plotDT_scurve = self.eval_list["plotDT_scurve"].merge(
            constr_labels, on="channel"
        )

        # 3. Process mainPoints data
        mainPoints = self.eval_list["mainPoints"].merge(constr_labels, on="channel")
        mainPoints = mainPoints.merge(
            self.resp_metric[["type", "type_lab"]], on="type", how="left"
        )

        # Handle type column first (matching R's mutate)
        mainPoints["type"] = mainPoints["type"].astype(str)
        mainPoints["type"] = pd.Categorical(
            mainPoints["type"].fillna("Carryover"),
            categories=["Carryover"] + list(self.resp_metric["type"].unique()),
        )

        # Handle type_lab column (matching R's mutate)
        mainPoints["type_lab"] = mainPoints["type_lab"].astype(str)
        mainPoints["type_lab"] = pd.Categorical(
            mainPoints["type_lab"].fillna("Carryover"),
            categories=["Carryover"] + list(self.resp_metric["type_lab"].unique()),
        )
        # Get carryover points
        caov_points = mainPoints[mainPoints["type"] == "Carryover"][
            ["channel", "spend_point"]
        ].rename(columns={"spend_point": "caov_spend"})

        # Merge and calculate constraint bounds
        mainPoints = mainPoints.merge(caov_points, on="channel")

        # Get the levels from resp_metric
        levs1 = self.resp_metric[
            "type"
        ].unique()  # Should contain ["Initial", "Unbounded"]

        # Calculate constraint bounds directly using pandas operations
        mainPoints["constr_low_abs"] = np.where(
            mainPoints["type"] == levs1[1],  # levs1[1] should be "Initial"
            mainPoints["constr_low_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_up_abs"] = np.where(
            mainPoints["type"] == levs1[1],
            mainPoints["constr_up_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_low_unb_abs"] = np.where(
            mainPoints["type"] == levs1[2],  # levs1[2] should be "Unbounded"
            mainPoints["constr_low_unb_abs"] + mainPoints["caov_spend"],
            np.nan,
        )
        mainPoints["constr_up_unb_abs"] = np.where(
            mainPoints["type"] == levs1[2],
            mainPoints["constr_up_unb_abs"] + mainPoints["caov_spend"],
            np.nan,
        )

        # Calculate plot bounds
        mainPoints["plot_lb"] = mainPoints["constr_low_abs"].fillna(
            mainPoints["constr_low_unb_abs"]
        )
        mainPoints["plot_ub"] = mainPoints["constr_up_abs"].fillna(
            mainPoints["constr_up_unb_abs"]
        )

        # 4. Create the plot with improved layout
        num_channels = len(plotDT_scurve["constr_label"].unique())
        num_rows = (num_channels + 2) // 3

        fig = make_subplots(
            rows=num_rows,
            cols=3,
            subplot_titles=plotDT_scurve["constr_label"].unique(),
            horizontal_spacing=0.15,  # Increased from 0.1
            vertical_spacing=0.25,  # Increased from 0.15
        )

        # Add traces for each channel
        for i, channel in enumerate(plotDT_scurve["constr_label"].unique()):
            row = (i // 3) + 1
            col = (i % 3) + 1

            channel_data = plotDT_scurve[plotDT_scurve["constr_label"] == channel]
            channel_points = mainPoints[mainPoints["constr_label"] == channel]

            # Carryover area first
            carryover_data = channel_data[
                channel_data["spend"] <= channel_data["mean_carryover"].iloc[0]
            ]
            if not carryover_data.empty:
                fig.add_trace(
                    go.Scatter(
                        x=carryover_data["spend"],
                        y=carryover_data["total_response"],
                        fill="tozeroy",
                        fillcolor="rgba(128, 128, 128, 0.4)",
                        mode="none",  # Changed from line=dict(width=0)
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

            # Response curve second
            fig.add_trace(
                go.Scatter(
                    x=channel_data["spend"],
                    y=channel_data["total_response"],
                    mode="lines",
                    name=channel,
                    line=dict(width=0.5),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

            # Add points and error bars
            if not channel_points.empty:
                # Get the bound multiplier from the data
                bound_mult = self.dt_optim_out["unconstr_mult"].iloc[0]

                # Create color mapping dictionary with dynamic bounded multiplier
                color_map = {
                    "Carryover": "white",
                    "Initial": "grey",
                    "Bounded": "steelblue",
                    f"Bounded x{bound_mult}": "darkgoldenrod",
                }

                # Add points for each type (only add to legend for first subplot)
                for type_label in color_map.keys():
                    # Use type column for Carryover, type_lab for others
                    if type_label == "Carryover":
                        type_points = channel_points[
                            channel_points["type"] == "Carryover"
                        ]
                    else:
                        type_points = channel_points[
                            channel_points["type_lab"] == type_label
                        ]

                    if not type_points.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=type_points["spend_point"],
                                y=type_points["response_point"],
                                mode="markers",
                                marker=dict(
                                    size=10,
                                    color=color_map[type_label],
                                    line=dict(color="black", width=1),
                                ),
                                name=type_label,
                                legendgroup=type_label,
                                showlegend=(
                                    i == 0
                                ),  # Only show in legend for first subplot
                            ),
                            row=row,
                            col=col,
                        )

                # Add error bars only for Bounded and Bounded x{bound_mult} points
                bounded_points = channel_points[
                    channel_points["type_lab"].isin(
                        ["Bounded", f"Bounded x{bound_mult}"]
                    )
                ].copy()  # Add .copy() to avoid SettingWithCopyWarning

                if not bounded_points.empty:
                    # First add the dotted lines between bounds
                    for _, point in bounded_points.iterrows():
                        if pd.notna(point["plot_lb"]) and pd.notna(point["plot_ub"]):
                            fig.add_trace(
                                go.Scatter(
                                    x=[point["plot_lb"], point["plot_ub"]],
                                    y=[
                                        point["response_point"],
                                        point["response_point"],
                                    ],
                                    mode="lines",
                                    line=dict(color="black", width=1, dash="dot"),
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )

                    # Then add the triangular markers at the bounds
                    for bound, symbol in [
                        ("plot_lb", "triangle-left"),
                        ("plot_ub", "triangle-right"),
                    ]:
                        bound_points = bounded_points[pd.notna(bounded_points[bound])]
                        if not bound_points.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=bound_points[bound],
                                    y=bound_points["response_point"],
                                    mode="markers",
                                    marker=dict(
                                        symbol=symbol,
                                        size=8,
                                        color="black",
                                    ),
                                    showlegend=False,
                                ),
                                row=row,
                                col=col,
                            )

        # Update layout with improved formatting
        fig.update_layout(
            title={
                "text": (
                    f"Simulated Response Curves<br>"
                    f"<span style='font-size:10px'>"
                    f"Spend per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type} "
                    f"(grey area: mean historical carryover) | "
                    f"Response [{self.budget_allocator.mmm_data.mmmdata_spec.dep_var_type}]"
                    "</span>"
                ),
                "y": 0.95,
                "x": 0.02,
                "xanchor": "left",
                "yanchor": "top",
                "font": {"size": 12},
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=0.85,
                xanchor="left",
                x=1.05,
                font=dict(size=10),
            ),
            height=300 * num_rows,
            width=1000,
            template="plotly_white",
            margin=dict(t=80, b=80, l=120, r=50),  # Reduced top margin from 120 to 80
            annotations=[
                dict(
                    text=f"Spend** per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type}",
                    x=0.5,
                    y=-0.15,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=10),
                ),
                dict(
                    text=f"Total Response [{self.budget_allocator.mmm_data.mmmdata_spec.dep_var_type}]",
                    x=-0.08,  # Adjusted position
                    y=0.35,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    textangle=-90,
                    font=dict(size=10),
                ),
            ],
        )

        fig.update_xaxes(
            tickfont={"size": 8},
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        )

        fig.update_yaxes(
            tickfont={"size": 8},
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(128, 128, 128, 0.2)",
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(128, 128, 128, 0.2)",
        )

        # Add bold channel names as annotations beneath each subplot
        unique_labels = list(plotDT_scurve["constr_label"].unique())
        for i, label in enumerate(unique_labels, start=1):
            # Retrieve the channel name corresponding to this subplot label
            channel = constr_labels[constr_labels["constr_label"] == label][
                "channel"
            ].iloc[0]
            # Determine axis keys (first subplot keys do not have a number)
            xaxis_key = "xaxis" if i == 1 else f"xaxis{i}"
            yaxis_key = "yaxis" if i == 1 else f"yaxis{i}"
            # Get the domain for the subplot
            x_domain = fig.layout[xaxis_key].domain
            y_domain = fig.layout[yaxis_key].domain
            # Calculate the horizontal midpoint and a position just below the subplot
            x_mid = (x_domain[0] + x_domain[1]) / 2
            y_pos = y_domain[0] - 0.08 * (y_domain[1] - y_domain[0])
            fig.add_annotation(
                x=x_mid,
                y=y_pos,
                xref="paper",
                yref="paper",
                text=f"<b>{channel}</b>",
                showarrow=False,
                font=dict(size=10),
                xanchor="center",
                yanchor="top",
            )

        return fig

    def _plot_combined_response_curves(
        self, max_projection_spend: float = 50000000
    ) -> go.Figure:
        """
        Plot response curves for all channels on a single plot for comparison.

        Args:
            max_projection_spend: Maximum spend to project for any channel (default: 50,000,000)
        """
        logger.info("Creating combined response curves plot")
        try:
            # Create a plotly figure
            fig = go.Figure()

            # Set up colors for channels and scenarios
            # Using Plotly's default color sequence
            colors = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]

            scenario_markers = [
                "circle",
                "square",
                "triangle-up",
            ]  # Marker shapes for scenarios

            # Get scenarios from resp_metric
            levs1 = self.resp_metric["type"].unique()
            bound_mult = self.dt_optim_out["unconstr_mult"].iloc[0]

            scenario_colors = ["gray", "#4682B4", "#DAA520"]  # Match existing colors

            # Get response curve data
            plotDT_scurve = self.eval_list["plotDT_scurve"]
            mainPoints = self.eval_list["mainPoints"]

            # Add channels section header to legend (dummy trace)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=0, color="rgba(0,0,0,0)"),
                    name="<b>Channels</b>",
                    showlegend=True,
                )
            )

            # Process channels
            for i, channel in enumerate(self.dt_optim_out["channels"]):
                # Filter data for this channel
                channel_data = plotDT_scurve[plotDT_scurve["channel"] == channel]

                # Get initial spend for this channel and mean carryover
                initial_spend = self.dt_optim_out["initSpendUnit"].iloc[i]
                carryover = (
                    channel_data["mean_carryover"].iloc[0]
                    if "mean_carryover" in channel_data.columns
                    else 0
                )

                # Calculate projection limit
                projection_limit = min(initial_spend * 3, max_projection_spend)

                # Filter data for historical part (up to initial spend)
                historical_data = channel_data[channel_data["spend"] <= initial_spend]

                # Add trace for historical curve (solid line)
                fig.add_trace(
                    go.Scatter(
                        x=historical_data["spend"],
                        y=historical_data["total_response"],
                        name=channel,  # No prefix needed
                        line=dict(color=colors[i % len(colors)], width=2),
                        showlegend=True,
                    )
                )

                # Add carryover area if data is available
                if carryover > 0:
                    carryover_data = channel_data[channel_data["spend"] <= carryover]
                    if not carryover_data.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=carryover_data["spend"],
                                y=carryover_data["total_response"],
                                fill="tozeroy",
                                fillcolor="rgba(128, 128, 128, 0.4)",
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo="none",
                            )
                        )

                # Project additional data if needed
                projection_data = channel_data[
                    (channel_data["spend"] > initial_spend)
                    & (channel_data["spend"] <= projection_limit)
                ]

                if (
                    projection_data.empty
                    and max(channel_data["spend"]) < projection_limit
                ):
                    # Need to extrapolate - use existing points to estimate extended curve
                    last_points = channel_data.nlargest(5, "spend")
                    max_response = last_points["total_response"].max()

                    # Create projection data points
                    projection_spend = np.linspace(
                        max(channel_data["spend"]), projection_limit, 50
                    )

                    # Simple saturation curve extrapolation (Hill-like behavior)
                    alpha = 0.7  # Typical value from Hill parameters
                    gamma = initial_spend  # Use initial spend as inflection point

                    # Hill transformation for extrapolation
                    projection_response = max_response * (
                        (projection_spend**alpha)
                        / ((projection_spend**alpha) + (gamma**alpha))
                    )

                    # Add trace for projection curve (dotted line)
                    fig.add_trace(
                        go.Scatter(
                            x=projection_spend,
                            y=projection_response,
                            name=f"{channel} (projected)",
                            line=dict(
                                color=colors[i % len(colors)], width=2, dash="dot"
                            ),
                            showlegend=False,
                        )
                    )
                else:
                    # Use existing data for projection
                    fig.add_trace(
                        go.Scatter(
                            x=projection_data["spend"],
                            y=projection_data["total_response"],
                            name=f"{channel} (projected)",
                            line=dict(
                                color=colors[i % len(colors)], width=2, dash="dot"
                            ),
                            showlegend=False,
                        )
                    )

            # Add scenarios section header to legend (dummy trace)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=0, color="rgba(0,0,0,0)"),
                    name="<b>Scenarios</b>",
                    showlegend=True,
                )
            )

            # Add scenario points - group together by scenario rather than by channel
            for scenario_idx, scenario_type in enumerate(levs1[1:], 1):
                # Only add to legend once per scenario
                fig.add_trace(
                    go.Scatter(
                        x=[None],  # Empty trace for legend only
                        y=[None],
                        mode="markers",
                        marker=dict(
                            color="black",  # Use black for legend item
                            size=10,
                            symbol=scenario_markers[
                                (scenario_idx - 1) % len(scenario_markers)
                            ],
                            line=dict(
                                color=scenario_colors[
                                    (scenario_idx - 1) % len(scenario_colors)
                                ],
                                width=2,
                            ),
                        ),
                        name=scenario_type,
                        showlegend=True,
                    )
                )

                # Add the actual scenario points for each channel (but don't add to legend)
                for i, channel in enumerate(self.dt_optim_out["channels"]):
                    scenario_points = mainPoints[
                        (mainPoints["channel"] == channel)
                        & (mainPoints["type"] == scenario_type)
                    ]

                    if not scenario_points.empty:
                        fig.add_trace(
                            go.Scatter(
                                x=scenario_points["spend_point"],
                                y=scenario_points["response_point"],
                                mode="markers",
                                marker=dict(
                                    color=colors[i % len(colors)],
                                    size=10,
                                    symbol=scenario_markers[
                                        (scenario_idx - 1) % len(scenario_markers)
                                    ],
                                    line=dict(
                                        color=scenario_colors[
                                            (scenario_idx - 1) % len(scenario_colors)
                                        ],
                                        width=2,
                                    ),
                                ),
                                showlegend=False,  # Don't show in legend
                            )
                        )

            # Add information section header to legend (dummy trace)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=0, color="rgba(0,0,0,0)"),
                    name="<b>Information</b>",
                    showlegend=True,
                )
            )

            # Add a trace for the projection line legend
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="lines",
                    line=dict(color="black", width=2, dash="dot"),
                    name="Projection (3x spend)",
                    showlegend=True,
                )
            )

            # Add a trace for carryover area legend
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="none",
                    fill="tozeroy",
                    fillcolor="rgba(128, 128, 128, 0.4)",
                    name="Historical Carryover",
                    showlegend=True,
                )
            )

            # Update layout
            fig.update_layout(
                title={
                    "text": (
                        f"Comparative Response Curves Across All Channels<br>"
                        f"<span style='font-size:10px'>"
                        f"Solid: Historical Data | Dotted: Projections (up to 3x initial spend) | "
                        f"Grey Area: Historical Carryover | "
                        f"Spend per {self.budget_allocator.mmm_data.mmmdata_spec.interval_type}"
                        "</span>"
                    ),
                    "y": 0.95,
                    "x": 0.02,
                    "xanchor": "left",
                    "yanchor": "top",
                    "font": {"size": 12},
                },
                xaxis_title="Spend",
                yaxis_title=f"Response [{self.budget_allocator.mmm_data.mmmdata_spec.dep_var_type}]",
                # Use a vertical legend on the right side with organized sections
                legend=dict(
                    orientation="v",  # Vertical orientation
                    yanchor="top",
                    y=1.0,
                    xanchor="left",
                    x=1.02,
                    font=dict(size=9),
                    itemsizing="constant",  # Make all legend icons the same size
                ),
                height=600,
                width=1000,
                template="plotly_white",
                margin=dict(
                    t=100, b=80, l=120, r=170
                ),  # Increased right margin for the legend
            )

            # Format axis labels with commas and limit x-axis to max_projection_spend
            fig.update_xaxes(
                tickformat=",",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.2)",
                range=[
                    0,
                    max_projection_spend,
                ],  # Set X-axis limit to max_projection_spend
            )

            fig.update_yaxes(
                tickformat=",",
                showgrid=True,
                gridwidth=1,
                gridcolor="rgba(128, 128, 128, 0.2)",
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor="rgba(128, 128, 128, 0.2)",
            )

            return fig

        except Exception as e:
            logger.error("Failed to create combined response curves plot: %s", str(e))
            raise

    def plot_all(
        self,
        display_plots: bool = True,
        export_location: Union[str, Path] = None,
        quiet: bool = True,
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
                "budget_opt": self._plot_response_spend_comparison(),
                "allocation": self._plot_allocation_comparison(),
                "response": self._plot_response_curves(),
                "combined_response": self._plot_combined_response_curves(
                    max_projection_spend=50000000
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
