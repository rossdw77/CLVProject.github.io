"""
statistical_models.py
---------------------
Probabilistic CLV models:
  1. BG/NBD + Gamma-Gamma via the lifetimes library
  2. BG/NBD + Gamma-Gamma via PyMC Marketing (Bayesian MCMC)

Both approaches follow the same pattern:
  - fit on training RFM data
  - predict expected purchases & average spend
  - compute CLV and evaluate vs. actuals
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

from src.config import (
    CUSTOMER_ID_COL, DATE_COL, AMOUNT_COL,
    BGNBD_PENALIZER, GGF_PENALIZER,
    CLTV_TIME_MONTHS, CLTV_FREQ, CLTV_DISCOUNT_RATE,
)


# ---------------------------------------------------------------------------
# lifetimes: BG/NBD
# ---------------------------------------------------------------------------

def fit_bgnbd(nbd_data: pd.DataFrame, penalizer_coef: float = BGNBD_PENALIZER):
    """
    Fit a BetaGeoFitter (BG/NBD) model.

    Parameters
    ----------
    nbd_data : DataFrame with columns Frequency, Recency, tenure
               (customers with Frequency >= 1 only).

    Returns
    -------
    Fitted BetaGeoFitter instance.
    """
    from lifetimes import BetaGeoFitter
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)
    bgf.fit(nbd_data["Frequency"], nbd_data["Recency"], nbd_data["tenure"])
    return bgf


def predict_expected_purchases(bgf, nbd_data: pd.DataFrame, weeks: int) -> pd.Series:
    """Predict expected number of purchases over `weeks` periods."""
    return bgf.predict(
        weeks,
        nbd_data["Frequency"],
        nbd_data["Recency"],
        nbd_data["tenure"],
    )


def plot_period_transactions(bgf) -> None:
    """Plot actual vs model-expected transaction counts per time period."""
    from lifetimes.plotting import plot_period_transactions
    import matplotlib.pyplot as plt
    plot_period_transactions(bgf)
    plt.show()


# ---------------------------------------------------------------------------
# lifetimes: Gamma-Gamma
# ---------------------------------------------------------------------------

def fit_gamma_gamma(nbd_data: pd.DataFrame, penalizer_coef: float = GGF_PENALIZER):
    """
    Fit a GammaGammaFitter model.
    Requires customers with Frequency >= 1 (repeat purchasers only).

    Returns
    -------
    Fitted GammaGammaFitter instance.
    """
    from lifetimes import GammaGammaFitter
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)
    ggf.fit(nbd_data["Frequency"], nbd_data[AMOUNT_COL])
    return ggf


def compute_cltv(
    bgf,
    ggf,
    nbd_data: pd.DataFrame,
    time: int = CLTV_TIME_MONTHS,
    freq: str = CLTV_FREQ,
    discount_rate: float = CLTV_DISCOUNT_RATE,
) -> pd.DataFrame:
    """
    Compute CLV using the fitted BG/NBD and Gamma-Gamma models.

    Returns
    -------
    DataFrame with customer_id and clv column.
    """
    cltv = ggf.customer_lifetime_value(
        bgf,
        nbd_data["Frequency"],
        nbd_data["Recency"],
        nbd_data["tenure"],
        nbd_data[AMOUNT_COL],
        time=time,
        freq=freq,
        discount_rate=discount_rate,
    )
    return cltv.reset_index()


def evaluate_cltv(
    nbd_data: pd.DataFrame,
    cltv_df: pd.DataFrame,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Merge CLV predictions with actuals, compute MAPE, and return decile metrics.

    Returns
    -------
    Tuple of (cltv_final DataFrame, decile_metrics DataFrame).
    """
    merged = pd.merge(
        nbd_data,
        cltv_df,
        left_on=nbd_data.index,
        right_on="index",
        how="left",
    )
    merged["Error"] = merged["clv"] - merged[AMOUNT_COL]
    merged["MAPE"] = abs(merged[AMOUNT_COL] - merged["clv"]) / merged[AMOUNT_COL]

    merged["decile"] = pd.qcut(merged[AMOUNT_COL], q=n_deciles, labels=False) + 1
    decile_metrics = (
        merged.groupby("decile")
        .apply(lambda g: pd.Series({
            "n": len(g),
            "actual_mean": g[AMOUNT_COL].mean(),
            "mae": mean_absolute_error(g[AMOUNT_COL], g["clv"]),
            "mape": mean_absolute_percentage_error(g[AMOUNT_COL], g["clv"]),
        }))
        .reset_index()
    )
    return merged, decile_metrics


# ---------------------------------------------------------------------------
# PyMC Marketing: BG/NBD (Bayesian)
# ---------------------------------------------------------------------------

def build_pymc_rfm_summary(
    train_btyd: pd.DataFrame,
    customer_id_col: str = CUSTOMER_ID_COL,
    datetime_col: str = DATE_COL,
    monetary_col: str = AMOUNT_COL,
    time_unit: str = "D",
):
    """
    Build the RFM summary DataFrame required by pymc_marketing.
    Note: frequency here is REPEAT transactions (first excluded per BTYD convention).
    """
    from pymc_marketing import clv
    rfm = clv.utils.rfm_summary(
        train_btyd,
        customer_id_col=customer_id_col,
        datetime_col=datetime_col,
        monetary_value_col=monetary_col,
        datetime_format="%Y-%m-%d",
        time_unit=time_unit,
    )
    return rfm


def fit_pymc_bgnbd(pymc_rfm_data: pd.DataFrame):
    """
    Fit a Bayesian BG/NBD model using PyMC Marketing.
    Uses Weibull priors (alpha, r) from the notebook.

    Returns
    -------
    Fitted BetaGeoModel instance with idata attached.
    """
    from pymc_marketing import clv
    from pymc_marketing.prior import Prior

    model = clv.BetaGeoModel(
        data=pymc_rfm_data,
        model_config={
            "alpha": Prior("Weibull", alpha=2, beta=200),
            "r": Prior("Weibull", alpha=2, beta=1),
        },
    )
    model.build_model()
    model.fit()
    return model


def fit_pymc_gamma_gamma(pymc_rfm_data: pd.DataFrame):
    """
    Fit a Bayesian Gamma-Gamma model using PyMC Marketing.
    Filters to repeat purchasers (frequency > 0).

    Returns
    -------
    Fitted GammaGammaModel instance.
    """
    from pymc_marketing import clv

    nonzero = pymc_rfm_data.query("frequency > 0")
    dataset = pd.DataFrame({
        "customer_id": nonzero.index,
        "monetary_value": nonzero["monetary_value"],
        "frequency": nonzero["frequency"],
    })

    model = clv.GammaGammaModel(data=dataset)
    model.build_model()
    model.fit()
    return model


def evaluate_pymc_models(
    bgm: object,
    gg: object,
    pymc_rfm_data: pd.DataFrame,
    test_btyd: pd.DataFrame,
    future_t: int = 180,
) -> pd.DataFrame:
    """
    Compute prob alive, expected purchases, and expected spend for the test period.
    Returns a merged sdata DataFrame with actuals for comparison.

    Parameters
    ----------
    bgm        : fitted BetaGeoModel
    gg         : fitted GammaGammaModel
    pymc_rfm_data : RFM summary for training period
    test_btyd  : raw test transactions
    future_t   : prediction horizon in days
    """
    from pymc_marketing import clv

    sdata = pymc_rfm_data.copy()

    # Build test-period RFM to get actual repeat purchases
    full_rfm = clv.utils.rfm_summary(
        test_btyd,
        customer_id_col=CUSTOMER_ID_COL,
        datetime_col=DATE_COL,
        monetary_value_col=AMOUNT_COL,
        datetime_format="%Y-%m-%d",
        time_unit="D",
    )
    full_rfm = full_rfm[["frequency"]].rename(columns={"frequency": "frequency_test"})
    sdata = sdata.join(full_rfm)

    # Posterior predictions
    pred_data = pymc_rfm_data[["customer_id", "frequency", "recency", "T"]]
    spend_data = pymc_rfm_data[["customer_id", "frequency", "monetary_value"]]

    prob_alive = bgm.expected_probability_alive(data=pred_data)
    expected_purchases = bgm.expected_purchases(data=pred_data, future_t=future_t)
    expected_spend = gg.expected_customer_spend(data=spend_data)

    sdata["prob_alive"] = prob_alive.mean(("chain", "draw")).values
    sdata["expected_purchases"] = expected_purchases.mean(("chain", "draw")).values.round(2)
    sdata["actual_purchases"] = (sdata["frequency_test"] - sdata["frequency"]).clip(lower=0)
    sdata["expected_monetary_value"] = expected_spend.mean(("chain", "draw")).values.round(2)
    sdata = sdata.fillna(0)

    return sdata


def print_pymc_metrics(sdata: pd.DataFrame) -> None:
    """Print RMSE and MAE for frequency and monetary value predictions."""
    freq_actual = sdata["actual_purchases"].tolist()
    freq_pred = sdata["expected_purchases"].tolist()
    mon_actual = sdata["monetary_value"].tolist()
    mon_pred = sdata["expected_monetary_value"].tolist()

    print(f"Frequency  — RMSE: {root_mean_squared_error(freq_actual, freq_pred):.4f}  "
          f"MAE: {mean_absolute_error(freq_actual, freq_pred):.4f}")
    print(f"Monetary   — RMSE: {root_mean_squared_error(mon_actual, mon_pred):.4f}  "
          f"MAE: {mean_absolute_error(mon_actual, mon_pred):.4f}")
