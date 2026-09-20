"""Visualization module producing publication-quality scientific figures."""
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent, clean scientific style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})
sns.set_theme(style="whitegrid", palette="deep")


def save_plot(fig: plt.Figure, output_path: Path, dpi: int = 300) -> None:
    """Save figure with high resolution and tight bounding box."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_yield_distribution(df: pd.DataFrame, output_path: Path) -> None:
    """Plot distribution of district-level rice yield across India."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["RICE.YIELD..Kg.per.ha."], kde=True, ax=ax, color="#1f77b4", bins=40)
    ax.axvline(df["RICE.YIELD..Kg.per.ha."].median(), color="red", linestyle="--", label=f"Median: {df['RICE.YIELD..Kg.per.ha.'].median():.0f} kg/ha")
    ax.axvline(df["RICE.YIELD..Kg.per.ha."].mean(), color="black", linestyle=":", label=f"Mean: {df['RICE.YIELD..Kg.per.ha.'].mean():.0f} kg/ha")
    ax.set_title("Distribution of Historical Rice Yield (1990–2017)")
    ax.set_xlabel("Rice Yield (kg/ha)")
    ax.set_ylabel("Observation Count")
    ax.legend(loc="upper right")
    save_plot(fig, output_path)


def plot_yield_over_time(df: pd.DataFrame, output_path: Path) -> None:
    """Plot national yearly mean and interquartile rice yield over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    yearly = df.groupby("Year")["RICE.YIELD..Kg.per.ha."].agg(
        mean="mean",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
    ).reset_index()

    ax.plot(yearly["Year"], yearly["mean"], color="#1f77b4", marker="o", linewidth=2, label="National Mean Yield")
    ax.fill_between(yearly["Year"], yearly["p25"], yearly["p75"], color="#1f77b4", alpha=0.2, label="Interquartile Range (25th–75th)")
    ax.set_title("District Rice Yield Trend Across India (1990–2017)")
    ax.set_xlabel("Crop Year")
    ax.set_ylabel("Rice Yield (kg/ha)")
    ax.set_xticks(yearly["Year"][::2])
    ax.legend(loc="upper left")
    save_plot(fig, output_path)


def plot_rainfall_over_time(df: pd.DataFrame, output_path: Path) -> None:
    """Plot aggregate annual/monsoon rainfall trend over time."""
    fig, ax = plt.subplots(figsize=(10, 5))
    rf_yearly = df.groupby("Year")["IMD_RF"].agg(
        mean="mean",
        p25=lambda x: np.percentile(x, 25),
        p75=lambda x: np.percentile(x, 75),
    ).reset_index()

    ax.plot(rf_yearly["Year"], rf_yearly["mean"], color="#2ca02c", marker="s", linewidth=2, label="Mean Aggregate Rainfall")
    ax.fill_between(rf_yearly["Year"], rf_yearly["p25"], rf_yearly["p75"], color="#2ca02c", alpha=0.2, label="Interquartile Range (25th–75th)")
    ax.set_title("Annual Aggregate Rainfall Trend (IMD_RF, 1990–2017)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.set_xticks(rf_yearly["Year"][::2])
    ax.legend(loc="upper right")
    save_plot(fig, output_path)


def plot_rainfall_vs_yield(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot with trend line for rainfall vs rice yield."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["IMD_RF"], df["RICE.YIELD..Kg.per.ha."], alpha=0.25, color="#1f77b4", s=15, edgecolors="none")
    sns.regplot(
        data=df,
        x="IMD_RF",
        y="RICE.YIELD..Kg.per.ha.",
        scatter=False,
        ax=ax,
        color="#d62728",
        line_kws={"linewidth": 2, "label": "Linear Fit"},
    )
    ax.set_title("Aggregate Rainfall vs. Rice Yield")
    ax.set_xlabel("Aggregate Rainfall IMD_RF (mm)")
    ax.set_ylabel("Rice Yield (kg/ha)")
    ax.legend(loc="upper right")
    save_plot(fig, output_path)


def plot_temperature_vs_yield(df: pd.DataFrame, output_path: Path) -> None:
    """Scatter plot with trend line for maximum temperature vs rice yield."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["IMD_Tmax"], df["RICE.YIELD..Kg.per.ha."], alpha=0.25, color="#ff7f0e", s=15, edgecolors="none")
    sns.regplot(
        data=df,
        x="IMD_Tmax",
        y="RICE.YIELD..Kg.per.ha.",
        scatter=False,
        ax=ax,
        color="#d62728",
        line_kws={"linewidth": 2, "label": "Linear Fit"},
    )
    ax.set_title("Mean Maximum Temperature vs. Rice Yield")
    ax.set_xlabel("Mean Maximum Temperature IMD_Tmax (°C)")
    ax.set_ylabel("Rice Yield (kg/ha)")
    ax.legend(loc="upper right")
    save_plot(fig, output_path)


def plot_state_mean_yield(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of average rice yield by Indian state."""
    state_col = "State_Clean" if "State_Clean" in df.columns else "State"
    state_avg = df.groupby(state_col)["RICE.YIELD..Kg.per.ha."].mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=state_avg, x="RICE.YIELD..Kg.per.ha.", y=state_col, ax=ax, palette="mako")
    ax.set_title("Average Rice Yield by State (1990–2017)")
    ax.set_xlabel("Mean Rice Yield (kg/ha)")
    ax.set_ylabel("State")
    save_plot(fig, output_path)


def plot_district_mean_yield(df: pd.DataFrame, output_path: Path, top_n: int = 15) -> None:
    """Plot top and bottom district mean yields across India."""
    dist_col = "stable_geographic_id" if "stable_geographic_id" in df.columns else "District"
    dist_avg = df.groupby(dist_col)["RICE.YIELD..Kg.per.ha."].mean().sort_values().reset_index()

    extreme_dists = pd.concat([dist_avg.head(top_n), dist_avg.tail(top_n)])
    extreme_dists["category"] = ["Lowest Yield"] * top_n + ["Highest Yield"] * top_n

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(
        data=extreme_dists,
        x="RICE.YIELD..Kg.per.ha.",
        y=dist_col,
        hue="category",
        dodge=False,
        palette={"Lowest Yield": "#e74c3c", "Highest Yield": "#2ecc71"},
        ax=ax,
    )
    ax.set_title(f"Districts with Lowest and Highest Mean Rice Yield (Top/Bottom {top_n})")
    ax.set_xlabel("Mean Rice Yield (kg/ha)")
    ax.set_ylabel("District Identifier (State_DistCode)")
    ax.legend(title="Cohort", loc="lower right")
    save_plot(fig, output_path)


def plot_observations_by_year(df: pd.DataFrame, output_path: Path) -> None:
    """Plot observation count per calendar year."""
    yearly_counts = df["Year"].value_counts().sort_index().reset_index()
    yearly_counts.columns = ["Year", "Count"]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=yearly_counts, x="Year", y="Count", ax=ax, color="#3498db")
    ax.set_title("Dataset Temporal Coverage: Observation Count per Year (1990–2017)")
    ax.set_xlabel("Crop Year")
    ax.set_ylabel("Number of District Records")
    plt.xticks(rotation=45)
    save_plot(fig, output_path)


def plot_correlation_matrix(df: pd.DataFrame, output_path: Path) -> None:
    """Correlation heatmap of target and candidate meteorological variables."""
    cols = ["RICE.YIELD..Kg.per.ha.", "IMD_RF", "IMD_Tmax", "SD_RF_jun_nov"]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    ax.set_title("Correlation Matrix of Rice Yield and Climate Variables")
    save_plot(fig, output_path)


def plot_actual_vs_predicted(pred_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    """Scatter plot comparing actual vs predicted rice yield on test set."""
    fig, ax = plt.subplots(figsize=(7, 7))
    actuals = pred_df["Actual"]
    preds = pred_df["Predicted"]

    min_val = min(actuals.min(), preds.min()) - 100
    max_val = max(actuals.max(), preds.max()) + 100

    ax.scatter(actuals, preds, alpha=0.3, color="#2980b9", s=20, edgecolors="none")
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--", linewidth=1.5, label="1:1 Perfect Prediction")
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.set_title(f"{model_name}: Actual vs. Predicted Rice Yield (Test Set: 2014–2017)")
    ax.set_xlabel("Actual Rice Yield (kg/ha)")
    ax.set_ylabel("Predicted Rice Yield (kg/ha)")
    ax.legend(loc="upper left")
    save_plot(fig, output_path)


def plot_residual_distribution(pred_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    """Plot distribution of prediction residuals (Actual - Predicted)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    residuals = pred_df["Residual"]
    sns.histplot(residuals, kde=True, color="#8e44ad", ax=ax, bins=35)
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error")
    ax.axvline(residuals.mean(), color="black", linestyle=":", linewidth=1.5, label=f"Mean Residual: {residuals.mean():.1f} kg/ha")
    ax.set_title(f"{model_name}: Residual Distribution (Actual - Predicted)")
    ax.set_xlabel("Residual (kg/ha)")
    ax.set_ylabel("Observation Count")
    ax.legend(loc="upper right")
    save_plot(fig, output_path)


def plot_state_mae(state_metrics_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    """Plot test MAE across states."""
    sorted_df = state_metrics_df.sort_values(by="MAE", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(data=sorted_df, x="MAE", y="State", ax=ax, palette="rocket")
    ax.set_title(f"{model_name}: Test Set Mean Absolute Error (MAE) by State")
    ax.set_xlabel("Mean Absolute Error (kg/ha)")
    ax.set_ylabel("State")
    save_plot(fig, output_path)


def plot_district_mae_distribution(district_metrics_df: pd.DataFrame, model_name: str, output_path: Path) -> None:
    """Plot histogram / KDE of district-level test MAE."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(district_metrics_df["MAE"], kde=True, color="#d35400", ax=ax, bins=30)
    ax.axvline(district_metrics_df["MAE"].median(), color="black", linestyle="--", label=f"Median District MAE: {district_metrics_df['MAE'].median():.1f} kg/ha")
    ax.set_title(f"{model_name}: Distribution of District-Level Test MAE")
    ax.set_xlabel("Test MAE (kg/ha)")
    ax.set_ylabel("Number of Districts")
    ax.legend(loc="upper right")
    save_plot(fig, output_path)
