"""
config.py
---------
Central configuration: file paths, date constants, column lists, and model parameters.
Update these values here instead of hunting through multiple files.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")
PLOTS_DIR = os.path.join(OUTPUTS_DIR, "plots")

RAW_DATA_FILE = os.path.join(DATA_DIR, "ecommerce_customer_behavior_dataset_v2.csv")

# ---------------------------------------------------------------------------
# Train / test split date
# ---------------------------------------------------------------------------
TRAIN_CUTOFF = "2023-08-31"   # rows <= this date go to train, > go to test

# ---------------------------------------------------------------------------
# Column identifiers
# ---------------------------------------------------------------------------
CUSTOMER_ID_COL = "Customer_ID"
DATE_COL = "Date"
AMOUNT_COL = "Total_Amount"
QUANTITY_COL = "Quantity"
DISCOUNT_COL = "Discount_Amount"
AGE_COL = "Age"
GENDER_COL = "Gender"
PRODUCT_COL = "Product_Category"
RATING_COL = "Customer_Rating"

SESSION_COLS = ["Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days"]

# Final feature columns fed to ML models
FINAL_FEATURE_COLS = [
    CUSTOMER_ID_COL,
    "Recency",
    "duration",
    "Frequency",
    AMOUNT_COL,
    QUANTITY_COL,
    AGE_COL,
    "Beauty",
    "Books",
    "Electronics",
    "Fashion",
    "Food",
    "Home & Garden",
    "Sports",
    "Toys",
    "Female",
    "Male",
    "Other",
    "Session_Duration_Minutes_Frequency",
    "Pages_Viewed_Frequency",
    "Delivery_Time_Days_Frequency",
]

# Columns to drop when building X matrices
DROP_FOR_REGRESSION = [AMOUNT_COL, CUSTOMER_ID_COL, "Binned_Amount", "ranking"]
DROP_FOR_CLASSIFICATION = [AMOUNT_COL, CUSTOMER_ID_COL, "Binned_Amount", "ranking"]


#Columns for BG-NBD

FINAL_BG_NBD_COLS = ['Customer_ID','Recency','Frequency','Tenure']


# ---------------------------------------------------------------------------
# Spend binning (used for Binned_Amount and decile plots)
# ---------------------------------------------------------------------------
SPEND_BINS = [36.0663, 159.81, 439.8525, 1299.55, 3441.9775, 7298.629, 15077.5614]

# CLV ranking thresholds (derived from training data quantiles 33/66)
RANKING_LOW_QUANTILE = 0.33
RANKING_HIGH_QUANTILE = 0.66
RANKING_LABELS = ["low", "mid", "high"]

# ---------------------------------------------------------------------------
# BG/NBD + Gamma-Gamma parameters
# ---------------------------------------------------------------------------
BGNBD_PENALIZER = 0.001
GGF_PENALIZER = 0.01
CLTV_TIME_MONTHS = 6
CLTV_FREQ = "D"
CLTV_DISCOUNT_RATE = 0.01

# ---------------------------------------------------------------------------
# Random state (reproducibility)
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
