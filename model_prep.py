"""
model_prep.py
-------------
Feature selection, outlier capping, spend binning, CLV ranking, and
X / y matrix construction for both regression and classification tasks.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.config import (
    AMOUNT_COL, CUSTOMER_ID_COL,
    FINAL_FEATURE_COLS,
    DROP_FOR_REGRESSION, DROP_FOR_CLASSIFICATION,
    SPEND_BINS, RANKING_LOW_QUANTILE, RANKING_HIGH_QUANTILE, RANKING_LABELS,
)
from src.data_loader import replace_with_thresholds


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------

def select_final_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns used in modelling (drops leakage / helper cols)."""
    available = [c for c in FINAL_FEATURE_COLS if c in df.columns]
    return df[available].copy()


# ---------------------------------------------------------------------------
# Outlier capping
# ---------------------------------------------------------------------------

def apply_outlier_capping(df: pd.DataFrame, variable: str = AMOUNT_COL) -> pd.DataFrame:
    """Cap outliers for a given variable using 1/99th percentile thresholds."""
    return replace_with_thresholds(df, variable)


# ---------------------------------------------------------------------------
# Spend binning
# ---------------------------------------------------------------------------

def add_spend_bins(
    df: pd.DataFrame,
    bins: list = SPEND_BINS,
    col: str = AMOUNT_COL,
) -> pd.DataFrame:
    """Add a Binned_Amount column using fixed quantile-derived bin edges."""
    df = df.copy()
    df["Binned_Amount"] = pd.cut(df[col], bins=bins)
    return df


# ---------------------------------------------------------------------------
# CLV ranking (for classification target)
# ---------------------------------------------------------------------------

def compute_ranking_thresholds(df: pd.DataFrame, col: str = AMOUNT_COL) -> tuple:
    """Derive low / high thresholds from training data quantiles."""
    low_thresh = df[col].quantile(RANKING_LOW_QUANTILE)
    high_thresh = df[col].quantile(RANKING_HIGH_QUANTILE)
    return low_thresh, high_thresh


def add_clv_ranking(
    df: pd.DataFrame,
    low_thresh: float,
    high_thresh: float,
    col: str = AMOUNT_COL,
) -> pd.DataFrame:
    """
    Add a 'ranking' column (low / mid / high) based on spend thresholds.
    Always derive thresholds from training data and pass them to the test set.
    """
    df = df.copy()
    df["ranking"] = pd.cut(
        df[col],
        bins=[0, low_thresh, high_thresh, float("inf")],
        labels=RANKING_LABELS,
    )
    return df


# ---------------------------------------------------------------------------
# X / y splits
# ---------------------------------------------------------------------------

def get_regression_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Return (X_train, y_train, X_test, y_test) for regression.
    Drops Total_Amount, Customer_ID, and Binned_Amount from features.
    """
    drop = [c for c in DROP_FOR_REGRESSION if c in train_df.columns]

    X_train = train_df.drop(drop, axis=1)
    y_train = train_df[AMOUNT_COL]

    X_test = test_df.drop([c for c in drop if c in test_df.columns], axis=1)
    y_test = test_df[AMOUNT_COL]

    return X_train, y_train, X_test, y_test


def get_classification_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple:
    """
    Return (X_train, y_train_enc, X_test, y_test_enc, le) for classification.
    Encodes the 'ranking' label with LabelEncoder.
    """
    drop = [c for c in DROP_FOR_CLASSIFICATION if c in train_df.columns]

    X_train = train_df.drop(drop, axis=1)
    y_train_raw = train_df["ranking"]

    X_test = test_df.drop([c for c in drop if c in test_df.columns], axis=1)
    y_test_raw = test_df["ranking"]

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw)

    return X_train, y_train_enc, X_test, y_test_enc, le



def get_regression_splits_by_decile(train_df: pd.DataFrame, test_df: pd.DataFrame, bin_column: str, amount_column: str):
    binned_X_train = {}
    binned_y_train = {}
    binned_X_test = {}
    binned_y_test = {}

    #bin_ranges = [bin for bin in df[bin_column].unique()]

    binned_X_train = {i: train_df.loc[train_df[bin_column] == bin_val].drop(columns ={amount_column, 'Customer_ID', 'ranking',bin_column}) for i, bin_val in enumerate(sorted(train_df[bin_column].unique()))}
    binned_y_train = {i: train_df.loc[train_df[bin_column] == bin_val][amount_column] for i, bin_val in enumerate(sorted(train_df[bin_column].unique()))}

    binned_X_test = {i: test_df.loc[test_df[bin_column] == bin_val].drop(columns ={amount_column, 'Customer_ID', 'ranking',bin_column}) for i, bin_val in enumerate(sorted(test_df[bin_column].dropna().unique()))}
    binned_y_test = {i: test_df.loc[test_df[bin_column] == bin_val][amount_column] for i, bin_val in enumerate(sorted(test_df[bin_column].dropna().unique()))}

    return binned_X_train, binned_y_train, binned_X_test, binned_y_test


# ---------------------------------------------------------------------------
# Class-based regression splits (routes customers to per-tier models)
# ---------------------------------------------------------------------------

def split_train_by_rank(
    train_df: pd.DataFrame,
) -> tuple:
    """
    Split training data into per-rank X/y dicts using TRUE labels.
    Returns (X_train_dict, y_train_dict).
    """
    drop = [c for c in DROP_FOR_CLASSIFICATION if c in train_df.columns]
    X_train_dict = {}
    y_train_dict = {}

    for rank in train_df["ranking"].dropna().unique():
        subset = train_df.loc[train_df["ranking"] == rank]
        X_train_dict[rank] = subset.drop(drop, axis=1)
        y_train_dict[rank] = subset[AMOUNT_COL]

    return X_train_dict, y_train_dict


def split_test_by_predicted_rank(
    test_df: pd.DataFrame,
    classifier,
    X_test: pd.DataFrame,
    le: LabelEncoder,
) -> tuple:
    """
    Split test data using PREDICTED labels from the classifier.
    Returns (X_test_dict, y_test_dict).
    """
    predictions = classifier.predict(X_test)
    drop = [c for c in DROP_FOR_CLASSIFICATION if c in test_df.columns] 

    X_test_dict = {}
    y_test_dict = {}

    for rank in RANKING_LABELS:
        encoded = le.transform([rank])[0]
        idx = X_test.index[predictions == encoded]
        X_test_dict[rank] = X_test.loc[idx]
        y_test_dict[rank] = test_df.loc[test_df.index.isin(idx), AMOUNT_COL]

    return X_test_dict, y_test_dict
