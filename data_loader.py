"""
data_loader.py
--------------
Raw data ingestion and basic outlier utilities.
These functions are intentionally side-effect-free: they return new objects
and never mutate the input dataframe.
"""

import pandas as pd
from src.config import RAW_DATA_FILE, DATE_COL, RATING_COL


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str = RAW_DATA_FILE) -> pd.DataFrame:
    """Read the raw CSV and apply essential dtype fixes."""
    df = pd.read_csv(path)
    df = _fix_dtypes(df)
    return df


def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df[RATING_COL] = df[RATING_COL].astype(str)
    return df


# ---------------------------------------------------------------------------
# Data quality checks
# ---------------------------------------------------------------------------

def missing_values(df: pd.DataFrame) -> pd.Series:
    """Return a Series of columns that have at least one missing value."""
    counts = df.isna().sum()
    return counts[counts > 0]


def summarize(df: pd.DataFrame) -> None:
    """Print shape, dtypes and basic descriptive stats."""
    print(f"Shape: {df.shape}")
    print("\nDtypes:\n", df.dtypes)
    print("\nDescriptive stats:\n", df.describe(include="all").T)


# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

def outlier_thresholds(df: pd.DataFrame, variable: str) -> tuple:
    """
    Return (low_limit, up_limit) based on 1st and 99th percentiles.
    Used as a capping strategy rather than removal.
    """
    low_limit = df[variable].quantile(0.01)
    up_limit = df[variable].quantile(0.99)
    return low_limit, up_limit


def replace_with_thresholds(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """
    Cap values outside [low_limit, up_limit] in-place and return the df.
    Mutates the passed dataframe to mirror notebook behaviour.
    """
    low_limit, up_limit = outlier_thresholds(df, variable)
    df.loc[df[variable] < low_limit, variable] = low_limit
    df.loc[df[variable] > up_limit, variable] = up_limit
    return df

def summary_stats(df: pd.DataFrame) -> pd.DataFrame:

    return df.describe().T
