"""
evaluation.py
-------------
Model evaluation utilities: decile-level MAPE, SHAP value computation,
and misrouting analysis for the class-based regression pipeline.
All functions return DataFrames or print summaries — no side effects on models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Regression evaluation
# ---------------------------------------------------------------------------

def build_results_df(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    bins: pd.Series = None,
) -> pd.DataFrame:
    """
    Combine actuals, predictions, and optional spend-bin labels into one df.
    Adds an Error column (actual - predicted).
    """
    df = pd.DataFrame({"Actuals": y_actual.values, "Predictions": y_pred})
    if bins is not None:
        df["Bins"] = bins.values
    df["Error"] = df["Actuals"] - df["Predictions"]
    return df


def compute_decile_metrics(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Split predictions into n_deciles by actual spend and compute MAE/MAPE
    per decile.  High-value customers are in the top decile.
    """
    df = pd.DataFrame({"actual": y_actual.values, "predicted": y_pred})
    df["decile"] = pd.qcut(df["actual"], q=n_deciles, labels=False) + 1

    metrics = (
        df.groupby("decile")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "actual_mean": g["actual"].mean(),
            "mae": mean_absolute_error(g["actual"], g["predicted"]),
            "mape": mean_absolute_percentage_error(g["actual"], g["predicted"]),
        }))
        .reset_index()
    )
    return metrics


def print_regression_metrics(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    label: str = "",
) -> None:
    """Print MAE and MAPE for a single split."""
    mae = mean_absolute_error(y_actual, y_pred)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE:  ${mae:,.2f}")
    print(f"{prefix}MAPE: {mape:.2%}")


# ---------------------------------------------------------------------------
# Classification evaluation
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model,
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    le,
    label: str = "",
) -> None:
    """Print ROC-AUC, classification report, and confusion matrix."""
    proba = model.predict_proba(X)
    preds = model.predict(X)

    auc = roc_auc_score(y_encoded, proba, multi_class="ovr", average="macro")
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}ROC AUC: {auc:.4f}")
    print(f"\n{prefix}Classification Report:")
    print(classification_report(y_encoded, preds, target_names=le.classes_))
    print(f"{prefix}Confusion Matrix:")
    print(confusion_matrix(y_encoded, preds))


# ---------------------------------------------------------------------------
# Class-based regression routing analysis
# ---------------------------------------------------------------------------

def routing_mape_report(
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test_encoded: np.ndarray,
    classifier_predictions: np.ndarray,
    X_test_dict: dict,
    y_test_dict: dict,
    rank_models: dict,
    le,
) -> pd.DataFrame:
    """
    For each rank tier, report MAPE split by correctly-routed vs mis-routed
    customers (i.e. those whose predicted tier != true tier).
    """
    rows = []
    for rank in ["low", "mid", "high"]:
        encoded = le.transform([rank])[0]
        predicted_in_tier = X_test.index[classifier_predictions == encoded]
        true_labels_for_tier = y_test_encoded[classifier_predictions == encoded]
        correct_mask = true_labels_for_tier == encoded

        y_true = test_df.loc[predicted_in_tier, "Total_Amount"].values
        y_pred = rank_models[rank].predict(X_test_dict[rank])

        mape_overall = mean_absolute_percentage_error(y_true, y_pred)
        mape_correct = (
            mean_absolute_percentage_error(y_true[correct_mask], y_pred[correct_mask])
            if correct_mask.sum() > 0 else np.nan
        )
        mape_misrouted = (
            mean_absolute_percentage_error(y_true[~correct_mask], y_pred[~correct_mask])
            if (~correct_mask).sum() > 0 else np.nan
        )

        rows.append({
            "rank": rank,
            "n_total": len(predicted_in_tier),
            "n_correct": int(correct_mask.sum()),
            "n_misrouted": int((~correct_mask).sum()),
            "mape_overall": mape_overall,
            "mape_correct": mape_correct,
            "mape_misrouted": mape_misrouted,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame):
    """
    Compute SHAP values using TreeExplainer (optimised for tree-based models).
    Returns (explainer, shap_values).
    """
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def plot_shap_summary(shap_values, X: pd.DataFrame, save_path: str = None) -> None:
    import shap
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_importance(shap_values, X: pd.DataFrame, save_path: str = None) -> None:
    import shap
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall(
    explainer, shap_values, X: pd.DataFrame, sample_index: int = 0, save_path: str = None
) -> None:
    import shap
    explanation = shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value,
        data=X.iloc[sample_index],
        feature_names=X.columns.tolist(),
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False)
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


#----------------------------------------------------------------------------
#Xgboost tree display
#----------------------------------------------------------------------------

def visualize_bin_tree(models_by_bin, X_train_binned, tree_index=0, max_depth=3):
    """
    Visualize a single decision tree from an XGBoost model.

    Args:
        models_by_bin   : The trained XGBoost model or GridSearchCV object.
        X_train_binned  : DataFrame of training features.
        tree_index      : Which tree in the booster to plot (default: 0).
        max_depth       : How many levels deep to render (default: 3, keeps it readable)
    """
    model = models_by_bin
    # If models_by_bin is a GridSearchCV object, get the best estimator
    if hasattr(model, 'best_estimator_'):
        estimator = model.best_estimator_
    else:
        estimator = model

    # XGBoost has its own plotting function
    fig, ax = plt.subplots(figsize=(24, 10))
    xgboost.plot_tree(
        estimator,
        num_trees=tree_index,
        rankdir='LR', # 'LR' for left to right orientation
        ax=ax
        # Removed 'depth=max_depth' as it caused a graphviz parsing error
    )
    ax.set_title(f"XGBoost Tree #{tree_index}", fontsize=14)
    plt.tight_layout()
    #plt.savefig(f"xgboost_tree_idx{tree_index}.png", dpi=150, bbox_inches="tight")
    plt.show()
