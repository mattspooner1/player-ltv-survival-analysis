"""Visualization module for Player LTV Survival Analysis.

Provides reusable, publication-quality plotting functions with consistent
styling. All plots are designed for non-technical stakeholder audiences:
plain-language labels, annotations, and clear takeaways.

Style defaults:
    - Seaborn whitegrid theme
    - tab10 colour palette
    - 300 DPI PNG export
    - 12 pt base font size

Functions:
    setup_style: Apply project-wide matplotlib/seaborn style settings.
    export_figure: Save a figure at publication quality.
    plot_survival_curves_by_segment: KM-style curves with CI bands.
    plot_hazard_ratio_forest: Forest plot of Cox PH hazard ratios.
    plot_ltv_distribution_by_segment: Box + strip plots of pLTV.
    plot_roi_waterfall: Waterfall chart for business case ROI.
    plot_feature_importance_bar: Horizontal bar chart of top features.
    plot_retention_offer_roi: Before/after intervention comparison.
    plot_channel_budget_reallocation: Budget shift recommendation chart.
    plot_early_prediction_comparison: 7-day vs full model accuracy.
    plot_pltv_vs_actual_scatter: Predicted vs actual LTV scatter.
    plot_churn_risk_segments: Risk segmentation donut or bar chart.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project colour palette (tab10 subset with semantic meaning)
# ---------------------------------------------------------------------------
COLOURS = {
    "primary": "#1f77b4",  # Blue - main brand / primary metric
    "secondary": "#ff7f0e",  # Orange - comparison / secondary metric
    "positive": "#2ca02c",  # Green - good outcomes / gains
    "negative": "#d62728",  # Red - bad outcomes / losses
    "neutral": "#7f7f7f",  # Grey - baselines / context
    "highlight": "#9467bd",  # Purple - call-to-action / highlight
    "tier_ftp": "#1f77b4",  # Blue - Free-to-Play
    "tier_premium": "#ff7f0e",  # Orange - Premium
    "tier_vip": "#2ca02c",  # Green - VIP
}

TIER_COLOURS = {
    "Free-to-Play": COLOURS["tier_ftp"],
    "Premium": COLOURS["tier_premium"],
    "VIP": COLOURS["tier_vip"],
    "Basic": COLOURS["tier_ftp"],
    "Pro": COLOURS["tier_premium"],
    "Enterprise": COLOURS["tier_vip"],
}

CHANNEL_COLOURS = {
    "organic": "#2ca02c",
    "ads": "#d62728",
    "partner": "#ff7f0e",
    "event": "#9467bd",
    "other": "#7f7f7f",
}


# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------


def setup_style(config: Optional[dict] = None) -> None:
    """Apply project-wide matplotlib and seaborn style settings.

    Args:
        config: Optional config dict with visualization section.
            Falls back to project defaults if not provided.
    """
    viz_cfg = (config or {}).get("visualization", {})
    style = viz_cfg.get("style", "whitegrid")
    palette = viz_cfg.get("palette", "tab10")
    font_size = viz_cfg.get("font_size", 12)
    context = viz_cfg.get("context", "notebook")

    sns.set_theme(style=style, palette=palette, context=context)
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size + 2,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "figure.titlesize": font_size + 4,
            "figure.dpi": 100,
            "savefig.dpi": viz_cfg.get("figure_dpi", 300),
            "savefig.bbox": "tight",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    logger.info("Applied project visualization style: %s / %s", style, palette)


# ---------------------------------------------------------------------------
# Export helper
# ---------------------------------------------------------------------------


def export_figure(
    fig: plt.Figure,
    name: str,
    output_dir: str = "outputs/figures",
    dpi: int = 300,
    formats: Optional[list[str]] = None,
) -> list[str]:
    """Save a matplotlib figure at publication quality.

    Args:
        fig: Matplotlib Figure object.
        name: Base filename (without extension).
        output_dir: Directory for saved figures.
        dpi: Resolution in dots per inch.
        formats: List of file formats (default: ['png']).

    Returns:
        List of saved file paths.
    """
    if formats is None:
        formats = ["png"]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    for fmt in formats:
        file_path = out_path / f"{name}.{fmt}"
        fig.savefig(file_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        saved_paths.append(str(file_path))
        logger.info("Figure saved: %s", file_path)

    return saved_paths


# ---------------------------------------------------------------------------
# Survival curve plots
# ---------------------------------------------------------------------------


def plot_survival_curves_by_segment(
    time_grid: np.ndarray,
    survival_matrices: dict[str, np.ndarray],
    segment_name: str = "Segment",
    title: str = "Player Retention Curves by Segment",
    colours: Optional[dict[str, str]] = None,
    show_ci: bool = True,
    ci_alpha: float = 0.15,
    figsize: tuple[int, int] = (10, 6),
    annotation: Optional[str] = None,
) -> plt.Figure:
    """Plot Kaplan-Meier style survival curves with optional CI bands.

    Args:
        time_grid: Array of time points (days).
        survival_matrices: Dict mapping segment labels to arrays of shape
            (n_samples, n_times). Mean +/- 1 SE used for CI bands.
        segment_name: Label for the legend title.
        title: Plot title.
        colours: Dict mapping segment labels to hex colours.
        show_ci: Whether to show confidence interval bands.
        ci_alpha: Transparency of CI bands.
        figsize: Figure size in inches.
        annotation: Optional text annotation in the lower-left.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colours = colours or {}

    for label, surv_matrix in survival_matrices.items():
        colour = colours.get(label, None)
        mean_surv = surv_matrix.mean(axis=0)
        se_surv = surv_matrix.std(axis=0) / np.sqrt(surv_matrix.shape[0])

        ax.plot(
            time_grid,
            mean_surv,
            label=f"{label} (n={surv_matrix.shape[0]})",
            color=colour,
            linewidth=2,
        )
        if show_ci and surv_matrix.shape[0] > 1:
            ax.fill_between(
                time_grid,
                np.clip(mean_surv - 1.96 * se_surv, 0, 1),
                np.clip(mean_surv + 1.96 * se_surv, 0, 1),
                alpha=ci_alpha,
                color=colour,
            )

    ax.set_xlabel("Days Since Signup", fontweight="bold")
    ax.set_ylabel("Retention Probability", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(time_grid.min(), time_grid.max())
    ax.legend(title=segment_name, loc="lower left", framealpha=0.9)
    ax.axhline(y=0.5, color="grey", linestyle="--", alpha=0.4, label="_nolegend_")

    # Add 90-day and 180-day reference lines
    for ref_day in [90, 180]:
        if ref_day <= time_grid.max():
            ax.axvline(
                x=ref_day,
                color="grey",
                linestyle=":",
                alpha=0.3,
                label="_nolegend_",
            )
            ax.text(
                ref_day + 2,
                1.01,
                f"Day {ref_day}",
                fontsize=9,
                color="grey",
                ha="left",
            )

    if annotation:
        ax.text(
            0.02,
            0.02,
            annotation,
            transform=ax.transAxes,
            fontsize=9,
            fontstyle="italic",
            color="grey",
            verticalalignment="bottom",
        )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Hazard ratio forest plot
# ---------------------------------------------------------------------------


def plot_hazard_ratio_forest(
    hr_df: pd.DataFrame,
    title: str = "What Drives Player Churn? (Hazard Ratios)",
    figsize: tuple[int, int] = (10, 8),
    highlight_threshold: float = 1.0,
    top_n: Optional[int] = None,
    feature_labels: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Create a forest plot of Cox PH hazard ratios.

    HR > 1 increases churn risk (red), HR < 1 decreases risk (green).
    Business-friendly feature labels can be provided.

    Args:
        hr_df: DataFrame with columns 'feature' and 'hazard_ratio'.
        title: Plot title.
        figsize: Figure dimensions.
        highlight_threshold: Reference line value (typically 1.0).
        top_n: If set, show only top N features by absolute HR distance.
        feature_labels: Dict mapping internal feature names to display labels.

    Returns:
        Matplotlib Figure.
    """
    df = hr_df.copy()

    if top_n is not None:
        df["hr_distance"] = abs(df["hazard_ratio"] - highlight_threshold)
        df = df.nlargest(top_n, "hr_distance").drop(columns="hr_distance")

    # Sort by hazard ratio for visual clarity
    df = df.sort_values("hazard_ratio", ascending=True).reset_index(drop=True)

    # Apply business-friendly labels
    if feature_labels:
        df["display_label"] = df["feature"].map(feature_labels).fillna(df["feature"])
    else:
        df["display_label"] = df["feature"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=figsize)

    # Colour by direction
    bar_colours = [
        COLOURS["negative"] if hr > highlight_threshold else COLOURS["positive"]
        for hr in df["hazard_ratio"]
    ]

    ax.barh(
        df["display_label"],
        df["hazard_ratio"],
        color=bar_colours,
        edgecolor="white",
        height=0.6,
    )

    # Reference line at HR = 1
    ax.axvline(
        x=highlight_threshold,
        color="black",
        linestyle="-",
        linewidth=1.5,
        alpha=0.7,
    )

    # Annotate each bar with HR value
    for i, (_, row) in enumerate(df.iterrows()):
        hr = row["hazard_ratio"]
        offset = 0.02 if hr >= highlight_threshold else -0.02
        ha = "left" if hr >= highlight_threshold else "right"
        ax.text(
            hr + offset,
            i,
            f"{hr:.2f}",
            va="center",
            ha=ha,
            fontsize=10,
            fontweight="bold",
        )

    ax.set_xlabel("Hazard Ratio (HR > 1 = Higher Churn Risk)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)

    # Add interpretation guide
    ax.text(
        0.98,
        0.02,
        "GREEN = Reduces Churn    RED = Increases Churn",
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="bottom",
        fontstyle="italic",
        color="grey",
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# LTV distribution by segment
# ---------------------------------------------------------------------------


def plot_ltv_distribution_by_segment(
    df: pd.DataFrame,
    value_col: str = "pltv",
    segment_col: str = "plan_tier",
    title: str = "Predicted 180-Day Revenue by Player Tier",
    figsize: tuple[int, int] = (10, 6),
    colours: Optional[dict[str, str]] = None,
    currency: str = "GBP",
) -> plt.Figure:
    """Box + strip plot showing LTV distribution across segments.

    Args:
        df: DataFrame containing value and segment columns.
        value_col: Column name for the LTV metric.
        segment_col: Column name for the grouping segment.
        title: Plot title.
        figsize: Figure dimensions.
        colours: Dict mapping segment values to colours.
        currency: Currency label for axis.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    colours = colours or TIER_COLOURS

    order = df.groupby(segment_col)[value_col].median().sort_values().index.tolist()
    palette = [colours.get(seg, COLOURS["primary"]) for seg in order]

    sns.boxplot(
        data=df,
        x=segment_col,
        y=value_col,
        order=order,
        palette=palette,
        width=0.5,
        fliersize=3,
        ax=ax,
    )
    sns.stripplot(
        data=df,
        x=segment_col,
        y=value_col,
        order=order,
        palette=palette,
        alpha=0.3,
        size=4,
        jitter=0.2,
        ax=ax,
    )

    # Annotate medians
    for i, seg in enumerate(order):
        median_val = df[df[segment_col] == seg][value_col].median()
        count = len(df[df[segment_col] == seg])
        ax.text(
            i,
            median_val,
            f"  {currency} {median_val:,.0f}\n  (n={count})",
            fontsize=9,
            fontweight="bold",
            va="bottom",
        )

    ax.set_xlabel("")
    ax.set_ylabel(f"Predicted 180-Day Revenue ({currency})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{currency} {x:,.0f}")
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# ROI waterfall chart
# ---------------------------------------------------------------------------


def plot_roi_waterfall(
    items: list[tuple[str, float]],
    title: str = "Retention Offer ROI Analysis",
    figsize: tuple[int, int] = (10, 6),
    currency: str = "GBP",
) -> plt.Figure:
    """Create a waterfall chart showing costs, benefits, and net ROI.

    Args:
        items: List of (label, value) tuples. Positive = benefit,
            negative = cost. Last item treated as total.
        title: Plot title.
        figsize: Figure dimensions.
        currency: Currency label.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    labels = [item[0] for item in items]
    values = [item[1] for item in items]

    cumulative = 0.0
    bottoms = []
    bar_colours = []

    for i, val in enumerate(values):
        if i == len(values) - 1:
            # Total bar starts from 0
            bottoms.append(0)
            bar_colours.append(COLOURS["positive"] if val >= 0 else COLOURS["negative"])
        else:
            bottoms.append(cumulative if val >= 0 else cumulative + val)
            bar_colours.append(COLOURS["positive"] if val >= 0 else COLOURS["negative"])
            cumulative += val

    bars = ax.bar(
        labels,
        [abs(v) for v in values],
        bottom=bottoms,
        color=bar_colours,
        edgecolor="white",
        width=0.5,
    )

    # Annotate bars
    for bar, val in zip(bars, values):
        y_pos = bar.get_y() + bar.get_height() + max(abs(v) for v in values) * 0.01
        sign = "+" if val > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{sign}{currency} {val:,.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
            color=COLOURS["positive"] if val >= 0 else COLOURS["negative"],
        )

    ax.set_ylabel(f"Value ({currency})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{currency} {x:,.0f}")
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Feature importance bar chart
# ---------------------------------------------------------------------------


def plot_feature_importance_bar(
    fi_df: pd.DataFrame,
    title: str = "Top Features Predicting Player Lifetime Value",
    top_n: int = 15,
    figsize: tuple[int, int] = (10, 7),
    feature_labels: Optional[dict[str, str]] = None,
    colour: str = COLOURS["primary"],
) -> plt.Figure:
    """Horizontal bar chart of feature importances.

    Args:
        fi_df: DataFrame with 'feature' and 'importance_pct' columns.
        title: Plot title.
        top_n: Number of features to show.
        figsize: Figure dimensions.
        feature_labels: Dict mapping feature names to display labels.
        colour: Bar colour.

    Returns:
        Matplotlib Figure.
    """
    df = fi_df.head(top_n).copy()
    df = df.sort_values("importance_pct", ascending=True)

    if feature_labels:
        df["display_label"] = df["feature"].map(feature_labels).fillna(df["feature"])
    else:
        df["display_label"] = df["feature"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(df["display_label"], df["importance_pct"], color=colour, height=0.6)

    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(
            row["importance_pct"] + 0.3,
            i,
            f"{row['importance_pct']:.1f}%",
            va="center",
            fontweight="bold",
            fontsize=10,
        )

    ax.set_xlabel("Relative Importance (%)", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_xlim(0, df["importance_pct"].max() * 1.15)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Channel budget reallocation
# ---------------------------------------------------------------------------


def plot_channel_budget_reallocation(
    channel_df: pd.DataFrame,
    title: str = "UA Budget Reallocation Recommendation",
    figsize: tuple[int, int] = (12, 6),
    currency: str = "GBP",
) -> plt.Figure:
    """Side-by-side bar chart comparing current vs recommended budget.

    Args:
        channel_df: DataFrame with columns: channel, current_budget,
            recommended_budget, pltv_mean.
        title: Plot title.
        figsize: Figure dimensions.
        currency: Currency label.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    channels = channel_df["channel"].values
    x = np.arange(len(channels))
    width = 0.35

    # Left panel: budget comparison
    ax1.bar(
        x - width / 2,
        channel_df["current_budget"],
        width,
        label="Current Budget",
        color=COLOURS["neutral"],
        edgecolor="white",
    )
    ax1.bar(
        x + width / 2,
        channel_df["recommended_budget"],
        width,
        label="Recommended Budget",
        color=COLOURS["primary"],
        edgecolor="white",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(channels, rotation=45, ha="right")
    ax1.set_ylabel(f"Annual Budget ({currency})", fontweight="bold")
    ax1.set_title("Budget Allocation", fontweight="bold")
    ax1.legend()
    ax1.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{currency} {v:,.0f}")
    )

    # Right panel: pLTV by channel
    colours_list = [CHANNEL_COLOURS.get(ch, COLOURS["primary"]) for ch in channels]
    bars = ax2.bar(x, channel_df["pltv_mean"], color=colours_list, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=45, ha="right")
    ax2.set_ylabel(f"Mean Predicted LTV ({currency})", fontweight="bold")
    ax2.set_title("Mean pLTV by Acquisition Channel", fontweight="bold")

    for bar, val in zip(bars, channel_df["pltv_mean"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{currency} {val:,.0f}",
            ha="center",
            fontweight="bold",
            fontsize=10,
        )

    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda v, _: f"{currency} {v:,.0f}")
    )

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Retention offer ROI comparison
# ---------------------------------------------------------------------------


def plot_retention_offer_comparison(
    simulation: dict[str, float],
    title: str = "Retention Offer: Cost vs Expected Benefit",
    figsize: tuple[int, int] = (8, 5),
    currency: str = "GBP",
) -> plt.Figure:
    """Bar chart comparing intervention cost to expected revenue gain.

    Args:
        simulation: Dict from PLTVCalculator.simulate_retention_intervention.
        title: Plot title.
        figsize: Figure dimensions.
        currency: Currency label.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    categories = ["Offer Cost", "Expected Revenue\nfrom Saved Players", "Net Benefit"]
    values = [
        -simulation["total_cost_gbp"],
        simulation["expected_revenue_gbp"],
        simulation["net_benefit_gbp"],
    ]
    bar_colours = [COLOURS["negative"], COLOURS["positive"], COLOURS["highlight"]]

    bars = ax.bar(categories, values, color=bar_colours, width=0.5, edgecolor="white")

    for bar, val in zip(bars, values):
        y_pos = bar.get_height() if val >= 0 else bar.get_height()
        va = "bottom" if val >= 0 else "top"
        sign = "+" if val > 0 else ""
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y_pos,
            f"{sign}{currency} {val:,.0f}",
            ha="center",
            va=va,
            fontweight="bold",
            fontsize=11,
        )

    roi_pct = simulation["roi_pct"]
    ax.set_title(
        f"{title}\nROI: {roi_pct:.0f}%",
        fontweight="bold",
        pad=15,
    )
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_ylabel(f"Value ({currency})", fontweight="bold")

    # Add summary annotation
    ax.text(
        0.98,
        0.95,
        (
            f"Players targeted: {simulation['n_targeted']}\n"
            f"Players saved: {simulation['n_saved']}\n"
            f"Retention uplift: {simulation['retention_uplift_pct']:.0f}%"
        ),
        transform=ax.transAxes,
        fontsize=9,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Early prediction comparison
# ---------------------------------------------------------------------------


def plot_early_prediction_comparison(
    metrics: dict[str, dict[str, float]],
    title: str = "Early Prediction Accuracy: 7-Day vs Full Model",
    figsize: tuple[int, int] = (10, 6),
    currency: str = "GBP",
) -> plt.Figure:
    """Grouped bar chart comparing 7-day early model vs full model.

    Args:
        metrics: Dict like {'7-day model': {'rmse': ..., 'r_squared': ...},
                            'Full model': {'rmse': ..., 'r_squared': ...}}.
        title: Plot title.
        figsize: Figure dimensions.
        currency: Currency label.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    model_names = list(metrics.keys())
    x = np.arange(len(model_names))

    # RMSE comparison
    rmse_vals = [metrics[m].get("rmse", 0) for m in model_names]
    bars1 = ax1.bar(
        x,
        rmse_vals,
        color=[COLOURS["secondary"], COLOURS["primary"]],
        width=0.4,
        edgecolor="white",
    )
    for bar, val in zip(bars1, rmse_vals):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 10,
            f"{currency} {val:,.0f}",
            ha="center",
            fontweight="bold",
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel(f"RMSE ({currency})", fontweight="bold")
    ax1.set_title("Prediction Error (Lower = Better)", fontweight="bold")

    # R-squared comparison
    r2_vals = [metrics[m].get("r_squared", 0) for m in model_names]
    bars2 = ax2.bar(
        x,
        r2_vals,
        color=[COLOURS["secondary"], COLOURS["primary"]],
        width=0.4,
        edgecolor="white",
    )
    for bar, val in zip(bars2, r2_vals):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.2f}",
            ha="center",
            fontweight="bold",
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names)
    ax2.set_ylabel("R-squared", fontweight="bold")
    ax2.set_title("Variance Explained (Higher = Better)", fontweight="bold")

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# pLTV scatter plot
# ---------------------------------------------------------------------------


def plot_pltv_vs_actual_scatter(
    actual: np.ndarray,
    predicted: np.ndarray,
    title: str = "Predicted vs Actual 180-Day Revenue",
    figsize: tuple[int, int] = (8, 8),
    currency: str = "GBP",
    segment_labels: Optional[np.ndarray] = None,
    segment_colours: Optional[dict[str, str]] = None,
) -> plt.Figure:
    """Scatter plot of predicted vs actual LTV with perfect-prediction line.

    Args:
        actual: Array of actual revenue values.
        predicted: Array of predicted revenue values.
        title: Plot title.
        figsize: Figure dimensions.
        currency: Currency label.
        segment_labels: Optional array of segment labels for colouring.
        segment_colours: Dict mapping segment labels to colours.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if segment_labels is not None and segment_colours is not None:
        for seg in sorted(set(segment_labels)):
            mask = segment_labels == seg
            ax.scatter(
                actual[mask],
                predicted[mask],
                c=segment_colours.get(seg, COLOURS["primary"]),
                label=seg,
                alpha=0.6,
                s=40,
                edgecolors="white",
                linewidth=0.5,
            )
        ax.legend(title="Segment")
    else:
        ax.scatter(
            actual,
            predicted,
            c=COLOURS["primary"],
            alpha=0.5,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )

    # Perfect prediction line
    max_val = max(actual.max(), predicted.max())
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="Perfect Prediction")

    ax.set_xlabel(f"Actual 180-Day Revenue ({currency})", fontweight="bold")
    ax.set_ylabel(f"Predicted 180-Day Revenue ({currency})", fontweight="bold")
    ax.set_title(title, fontweight="bold", pad=15)

    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{currency} {x:,.0f}")
    )
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{currency} {x:,.0f}")
    )

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Churn risk segmentation
# ---------------------------------------------------------------------------


def plot_churn_risk_segments(
    risk_df: pd.DataFrame,
    title: str = "Player Churn Risk Distribution",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Stacked bar or grouped bar showing risk segments.

    Args:
        risk_df: DataFrame with 'risk_segment' and 'count' columns,
            plus optional 'mean_pltv' for annotation.
        title: Plot title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    risk_order = ["Low Risk", "Medium Risk", "High Risk"]
    risk_colours = [COLOURS["positive"], COLOURS["secondary"], COLOURS["negative"]]

    # Filter and order
    df = risk_df.set_index("risk_segment").reindex(risk_order).reset_index()

    # Left: count by segment
    ax1.bar(
        df["risk_segment"],
        df["count"],
        color=risk_colours,
        edgecolor="white",
    )
    for i, row in df.iterrows():
        ax1.text(
            i,
            row["count"] + 1,
            f"{row['count']}",
            ha="center",
            fontweight="bold",
        )
    ax1.set_ylabel("Number of Players", fontweight="bold")
    ax1.set_title("Players by Risk Segment", fontweight="bold")

    # Right: mean pLTV by segment
    if "mean_pltv" in df.columns:
        ax2.bar(
            df["risk_segment"],
            df["mean_pltv"],
            color=risk_colours,
            edgecolor="white",
        )
        for i, row in df.iterrows():
            ax2.text(
                i,
                row["mean_pltv"] + 10,
                f"GBP {row['mean_pltv']:,.0f}",
                ha="center",
                fontweight="bold",
            )
        ax2.set_ylabel("Mean Predicted LTV (GBP)", fontweight="bold")
        ax2.set_title("Mean pLTV by Risk Level", fontweight="bold")

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Summary KPI dashboard panel
# ---------------------------------------------------------------------------


def plot_kpi_summary(
    kpis: dict[str, Any],
    title: str = "Player LTV Model: Key Performance Indicators",
    figsize: tuple[int, int] = (14, 4),
) -> plt.Figure:
    """Create a simple KPI summary panel with large numbers.

    Args:
        kpis: Dict mapping KPI labels to values (str-formatted).
            Example: {'C-Index': '0.52', 'Mean pLTV': 'GBP 1,285',
                      'Players at Risk': '21 (20%)', 'ROI': '175%'}
        title: Panel title.
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    n_kpis = len(kpis)
    fig, axes = plt.subplots(1, n_kpis, figsize=figsize)
    if n_kpis == 1:
        axes = [axes]

    for ax, (label, value) in zip(axes, kpis.items()):
        ax.text(
            0.5,
            0.6,
            str(value),
            ha="center",
            va="center",
            fontsize=24,
            fontweight="bold",
            color=COLOURS["primary"],
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.2,
            label,
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

    fig.suptitle(title, fontweight="bold", fontsize=14, y=1.05)
    fig.tight_layout()
    return fig
