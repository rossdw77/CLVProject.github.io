"""
visualizations.py
-----------------
Reusable chart templates for EDA and model evaluation.
All functions accept a save_path argument; pass None to display inline.
Consistent figure size, color palette, and layout are set here so every
chart in the project looks the same.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------
FIGSIZE_WIDE = (15, 5)
FIGSIZE_SQUARE = (10, 10)
FIGSIZE_STANDARD = (10, 5)
PALETTE = "muted"


def _save_or_show(fig, save_path: str = None) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# EDA — distributions
# ---------------------------------------------------------------------------

def plot_distribution(
    series: pd.Series,
    title: str = "",
    plot_type: str = "hist",  # "hist" | "kde" | "box"
    figsize: tuple = FIGSIZE_STANDARD,
    save_path: str = None,
) -> None:
    """Single-variable distribution: histogram, KDE, or boxplot."""
    fig, ax = plt.subplots(figsize=figsize)
    if plot_type == "hist":
        sns.histplot(series, kde=True, ax=ax)
    elif plot_type == "kde":
        sns.kdeplot(series, ax=ax)
    elif plot_type == "box":
        sns.boxplot(series, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    _save_or_show(fig, save_path)


def plot_multi_distribution(
    df: pd.DataFrame,
    columns: list,
    plot_type: str = "hist",
    figsize: tuple = FIGSIZE_WIDE,
    save_path: str = None,
) -> None:
    """Grid of distribution plots for multiple columns."""
    n = len(columns)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, columns):
        if plot_type == "hist":
            sns.histplot(df[col], kde=True, ax=ax)
        elif plot_type == "kde":
            sns.kdeplot(df[col], ax=ax)
        elif plot_type == "box":
            sns.boxplot(df[col], ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# EDA — categorical and time-series
# ---------------------------------------------------------------------------

def plot_category_spend(
    df: pd.DataFrame,
    category_col: str,
    amount_col: str = "Total_Amount",
    plot_type: str = "bar",  # "bar" | "box" | "hist"
    figsize: tuple = FIGSIZE_WIDE,
    save_path: str = None,
) -> None:
    """Spend by product category."""
    fig, ax = plt.subplots(figsize=figsize)
    if plot_type == "bar":
        sns.barplot(x=category_col, y=amount_col, data=df, palette=PALETTE, ax=ax)
    elif plot_type == "box":
        sns.boxplot(x=category_col, y=amount_col, data=df, palette=PALETTE, ax=ax)
    elif plot_type == "hist":
        sns.histplot(x=amount_col, hue=category_col, data=df, ax=ax)
    ax.set_title(f"{amount_col} by {category_col}")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_time_series(
    df: pd.DataFrame,
    date_col: str,
    amount_col: str,
    rolling_window: int = 30,
    figsize: tuple = FIGSIZE_WIDE,
    save_path: str = None,
) -> None:
    """Daily spend with optional rolling mean overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(df[date_col], df[amount_col], alpha=0.4, label="Daily")
    if rolling_window:
        rolled = df[amount_col].rolling(rolling_window).mean()
        ax.plot(df[date_col], rolled, label=f"{rolling_window}-day avg", linewidth=2)
    ax.set_title(f"{amount_col} over time")
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# EDA — correlation
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(
    df: pd.DataFrame,
    figsize: tuple = FIGSIZE_SQUARE,
    save_path: str = None,
) -> None:
    """Correlation heatmap for numeric columns."""
    numeric = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_pairplot(
    df: pd.DataFrame,
    columns: list,
    hue: str = None,
    save_path: str = None,
) -> None:
    """Pairplot for selected columns."""
    g = sns.pairplot(df[columns + ([hue] if hue else [])], hue=hue, palette=PALETTE)
    if save_path:
        g.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    title: str = "Feature Importance",
    figsize: tuple = FIGSIZE_SQUARE,
    save_path: str = None,
) -> None:
    """
    Bar chart of feature importances.
    Expects a DataFrame with columns ['Feature', 'Importance'].
    """
    g = sns.catplot(
        data=importance_df,
        x="Feature",
        y="Importance",
        hue="Feature",
        legend=False,
        kind="bar",
        height=figsize[1],
        aspect=figsize[0] / figsize[1],
        palette=PALETTE,
    )
    for ax in g.axes.flatten():
        ax.tick_params(axis="x", rotation=90)
    g.figure.suptitle(title, y=1.01)
    plt.tight_layout()
    if save_path:
        g.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_decile_mape(
    decile_metrics: pd.DataFrame,
    title: str = "MAPE by Decile",
    figsize: tuple = FIGSIZE_STANDARD,
    save_path: str = None,
) -> None:
    """Bar chart of MAPE per spend decile."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x="decile", y="mape", data=decile_metrics, hue="decile", legend=False, palette=PALETTE, ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3)
    ax.set_title(title)
    ax.set_ylabel("MAPE")
    ax.set_xlabel("Spend Decile")
    plt.tight_layout()
    _save_or_show(fig, save_path)



def plot_actuals_vs_predictions(
    results_df: pd.DataFrame,
    title: str = "Actuals vs Predictions",
    figsize: tuple = FIGSIZE_STANDARD,
    add_regline: bool = True,
    save_path: str = None,
) -> None:
    """Scatter (+ optional regression line) of actuals vs predicted spend."""
    fig, ax = plt.subplots(figsize=figsize)
    if add_regline:
        sns.regplot(x="Actuals", y="Predictions", data=results_df, ax=ax, scatter_kws={"alpha": 0.3})
    else:
        sns.scatterplot(x="Actuals", y="Predictions", data=results_df, ax=ax, alpha=0.4)
    ax.set_title(title)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_error_distribution(
    results_df: pd.DataFrame,
    title: str = "Prediction Error Distribution",
    xlim: tuple = (-5000, 5000),
    figsize: tuple = FIGSIZE_STANDARD,
    save_path: str = None,
) -> None:
    """Histogram of prediction errors."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(x="Error", data=results_df, kde=True, ax=ax)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_roc_curves(
    y_true_encoded: np.ndarray,
    y_proba: np.ndarray,
    le,
    title: str = "ROC Curves",
    figsize: tuple = FIGSIZE_STANDARD,
    save_path: str = None,
) -> None:
    """One-vs-rest ROC curves for multi-class classification."""
    from sklearn.metrics import roc_curve
    from sklearn.preprocessing import label_binarize

    classes = le.classes_
    y_bin = label_binarize(y_true_encoded, classes=range(len(classes)))

    fig, ax = plt.subplots(figsize=figsize)
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, label=cls)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    _save_or_show(fig, save_path)


# ---------------------------------------------------------------------------
# BTYD / statistical model plots
# ---------------------------------------------------------------------------

def plot_rfm_histograms(
    rfm_df: pd.DataFrame,
    figsize: tuple = FIGSIZE_WIDE,
    save_path: str = None,
) -> None:
    """Histograms of recency, frequency, and monetary value from RFM summary."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    rfm_df["recency"].plot(kind="hist", bins=100, ax=axes[0], title="Recency")
    rfm_df["frequency"].plot(kind="hist", bins=100, ax=axes[1], title="Frequency")
    rfm_df["monetary_value"].plot(kind="hist", bins=100, ax=axes[2], title="Avg Monetary Value")
    plt.tight_layout()
    _save_or_show(fig, save_path)
