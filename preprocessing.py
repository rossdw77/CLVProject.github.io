"""
preprocessing.py
----------------
All customer-level aggregations, train/test date splits, and categorical
encoding.  Every function takes the cleaned int_df (from data_loader) and
a cutoff date, making the pipeline fully reproducible and easy to re-run
on a new snapshot.
"""

import numpy as np
import pandas as pd
from lifetimes.utils import summary_data_from_transaction_data

from src.config import (
    CUSTOMER_ID_COL, DATE_COL, AMOUNT_COL, QUANTITY_COL, DISCOUNT_COL,
    AGE_COL, GENDER_COL, PRODUCT_COL, SESSION_COLS, TRAIN_CUTOFF,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split(df: pd.DataFrame, cutoff: str, train: bool) -> pd.DataFrame:
    """Filter df to train (<=cutoff) or test (>cutoff) window."""
    mask = df[DATE_COL] <= cutoff if train else df[DATE_COL] > cutoff
    return df.loc[mask]


def _per_frequency(df: pd.DataFrame, freq_series: pd.Series) -> pd.DataFrame:
    """Divide every non-frequency column by transaction frequency."""
    result = df.copy()
    for col in result.columns:
        if col != "Frequency":
            result[f"{col}_Frequency"] = result[col] / freq_series
    return result


# ---------------------------------------------------------------------------
# Customer-level aggregations (full dataset — used for EDA)
# ---------------------------------------------------------------------------

def build_customer_purchases(df: pd.DataFrame) -> pd.DataFrame:
    """Recency, duration, tenure at the customer level."""
    agg = df.groupby(CUSTOMER_ID_COL)[DATE_COL].agg(["min", "max"])
    global_max = df[DATE_COL].max()

    agg["duration"] = (agg["max"] - agg["min"]).dt.days.astype(int)
    agg["tenure"] = (global_max - agg["min"]).dt.days.astype(int)
    agg["since_last_purchase"] = (global_max - agg["max"]).dt.days.astype(int)
    return agg


def build_customer_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Spend, quantity, discount totals + per-transaction ratios."""
    freq = df.groupby(CUSTOMER_ID_COL)[DATE_COL].nunique().rename("Frequency")
    trans = df.groupby(CUSTOMER_ID_COL)[[AMOUNT_COL, QUANTITY_COL, DISCOUNT_COL]].sum()
    merged = trans.join(freq)
    return _per_frequency(merged, merged["Frequency"])


def build_customer_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """Session duration, pages viewed, delivery time + per-transaction ratios."""
    freq = df.groupby(CUSTOMER_ID_COL)[DATE_COL].nunique().rename("Frequency")
    sess = df.groupby(CUSTOMER_ID_COL)[SESSION_COLS].sum()
    merged = sess.join(freq)
    return _per_frequency(merged, merged["Frequency"])


def build_customer_visits(df: pd.DataFrame) -> pd.DataFrame:
    """Visit-level metrics ranked by date per customer."""
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

    visits = (
        df.groupby(cat_cols)[num_cols]
        .sum()
        .reset_index()
        .sort_values([CUSTOMER_ID_COL, DATE_COL])
    )
    visits["Visit_Num"] = visits.groupby(CUSTOMER_ID_COL)[DATE_COL].rank()
    return visits


def build_multi_product_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Customers who bought more than one product category on the same date."""
    counts = (
        df.groupby([CUSTOMER_ID_COL, DATE_COL])[PRODUCT_COL]
        .nunique()
        .reset_index()
    )
    multi_dates = counts.loc[counts[PRODUCT_COL] > 1, [CUSTOMER_ID_COL, DATE_COL]]

    mask = df[CUSTOMER_ID_COL].isin(multi_dates[CUSTOMER_ID_COL]) & \
           df[DATE_COL].isin(multi_dates[DATE_COL])
    return df.loc[mask].sort_values([CUSTOMER_ID_COL, DATE_COL])


def build_single_product_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Customers who bought exactly one product category per visit."""
    multi = build_multi_product_customers(df)
    return df.loc[~df[CUSTOMER_ID_COL].isin(multi[CUSTOMER_ID_COL])]


# ---------------------------------------------------------------------------
# RFM dataset builders (windowed by cutoff date)
# ---------------------------------------------------------------------------

def _build_rfm_core(df: pd.DataFrame, cutoff: str, train: bool) -> pd.DataFrame:
    """Core RFM: Recency, Frequency, Total_Amount, Quantity, Discount, Age."""
    window = _split(df, cutoff, train)
    global_max = df[DATE_COL].max()

    purchases = window.groupby(CUSTOMER_ID_COL)[DATE_COL].agg(["min", "max"])
    purchases["duration"] = (purchases["max"] - purchases["min"]).dt.days.astype(int)
    purchases["tenure"] = (global_max - purchases["min"]).dt.days.astype(int)
    purchases["since_last_purchase"] = (global_max - purchases["max"]).dt.days.astype(int)
    purchases = purchases.reset_index()[
        [CUSTOMER_ID_COL, "since_last_purchase", "duration", "tenure"]
    ]

    freq = window.groupby(CUSTOMER_ID_COL)[DATE_COL].nunique().reset_index()
    freq.columns = [CUSTOMER_ID_COL, "Frequency"]

    trans = window.groupby(CUSTOMER_ID_COL).agg(
        {AMOUNT_COL: "sum", QUANTITY_COL: "sum", DISCOUNT_COL: "sum", AGE_COL: "max"}
    ).reset_index()

    rfm = (
        purchases
        .merge(freq, on=CUSTOMER_ID_COL)
        .merge(trans[[CUSTOMER_ID_COL, AMOUNT_COL, QUANTITY_COL, DISCOUNT_COL, AGE_COL]],
               on=CUSTOMER_ID_COL)
        .rename(columns={"since_last_purchase": "Recency"})
    )
    rfm["UpT"] = rfm[QUANTITY_COL] / rfm["Frequency"]
    rfm["DpT"] = rfm[DISCOUNT_COL] / rfm["Frequency"]
    return rfm


def _add_session_features(
    rfm: pd.DataFrame, df: pd.DataFrame, cutoff: str, train: bool
) -> pd.DataFrame:
    """Merge per-transaction session averages into the RFM frame."""
    window = _split(df, cutoff, train)
    freq = window.groupby(CUSTOMER_ID_COL)[DATE_COL].nunique().rename("Frequency")
    sess = window.groupby(CUSTOMER_ID_COL)[SESSION_COLS].sum().join(freq)
    sess_per_freq = _per_frequency(sess, sess["Frequency"])

    per_freq_cols = [f"{c}_Frequency" for c in SESSION_COLS]
    return rfm.merge(
        sess_per_freq.reset_index()[[CUSTOMER_ID_COL] + per_freq_cols],
        on=CUSTOMER_ID_COL,
    )


def _add_product_dummies(
    rfm: pd.DataFrame, df: pd.DataFrame, cutoff: str, train: bool
) -> pd.DataFrame:
    """One-hot encode Product_Category and merge counts per customer."""
    window = _split(df, cutoff, train)
    dummies = pd.get_dummies(window[PRODUCT_COL])
    cat = (
        pd.concat([window[CUSTOMER_ID_COL], dummies], axis=1)
        .groupby(CUSTOMER_ID_COL)
        .sum()
    )
    return rfm.merge(cat, on=CUSTOMER_ID_COL)


def _add_gender_dummies(
    rfm: pd.DataFrame, df: pd.DataFrame, cutoff: str, train: bool
) -> pd.DataFrame:
    """One-hot encode Gender and merge (min aggregation = stable single value)."""
    window = _split(df, cutoff, train)
    dummies = pd.get_dummies(window[GENDER_COL], dtype=int)
    gender = (
        pd.concat([window[CUSTOMER_ID_COL], dummies], axis=1)
        .groupby(CUSTOMER_ID_COL)
        .min()
        .reset_index()
    )
    return rfm.merge(gender, on=CUSTOMER_ID_COL)


def build_rfm_dataset(
    df: pd.DataFrame,
    cutoff: str = TRAIN_CUTOFF,
    train: bool = True,
) -> pd.DataFrame:
    """
    Full RFM dataset: core metrics + session features + product + gender dummies.

    Parameters
    ----------
    df     : cleaned int_df from data_loader
    cutoff : date string, e.g. '2023-08-31'
    train  : True → rows <= cutoff, False → rows > cutoff
    """
    rfm = _build_rfm_core(df, cutoff, train)
    rfm = _add_session_features(rfm, df, cutoff, train)
    rfm = _add_product_dummies(rfm, df, cutoff, train)
    rfm = _add_gender_dummies(rfm, df, cutoff, train)
    return rfm


# ---------------------------------------------------------------------------
# BTYD dataset (for lifetimes / PyMC Marketing)
# ---------------------------------------------------------------------------

def build_btyd_dataset(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF) -> pd.DataFrame:
    """
    Return (train_btyd, test_btyd) filtered by cutoff.
    Frequency >= 1 filter applied; first purchase excluded per BTYD convention.
    """
    train = df.loc[df[DATE_COL] <= cutoff].copy()
    test = df.loc[df[DATE_COL] > cutoff].copy()
    return train, test


# ---------------------------------------------------------------------------
# BG-NBD dataset
# ---------------------------------------------------------------------------

def build_bg_nbd_dataset(df: pd.DataFrame, cutoff: str = TRAIN_CUTOFF) -> pd.DataFrame:
    """
    Build the BTYD summary DataFrame for the training window using the lifetimes
    summary_data_from_transaction_data utility.
    Frequency >= 1 filter applied; first purchase excluded per BTYD convention.

    Returns
    -------
    DataFrame with columns: Customer_ID, Frequency, Recency, tenure, Total_Amount
    """

    #train = df.loc[df[DATE_COL] <= cutoff].copy()

    btyd = summary_data_from_transaction_data(
        train,
        #customer_id_col=CUSTOMER_ID_COL,
        datetime_col=DATE_COL,
        monetary_value_col=AMOUNT_COL,
        #observation_period_end=cutoff,
    )

    btyd = btyd.rename(columns={
        "frequency": "Frequency",
        "recency": "Recency",
        "T": "tenure",
        "monetary_value": AMOUNT_COL,
    })

    return btyd.loc[btyd["Frequency"] >= 1].reset_index()
