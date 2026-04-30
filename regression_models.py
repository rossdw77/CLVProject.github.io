"""
regression_models.py
--------------------
Five regression approaches for predicting total customer spend:
  1. Linear Regression (sklearn baseline)
  2. Random Forest (fixed params)
  3. XGBoost Simple (no tuning — quick benchmark)
  4. XGBoost Tuned (RandomizedSearchCV)
  5. Class-based XGBoost (one model per CLV tier)

All train functions return the fitted model / search object.
Use evaluation.compute_decile_metrics() and evaluation.build_results_df()
for post-fit analysis.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor

from src.config import RANDOM_STATE, RANKING_LABELS


# ---------------------------------------------------------------------------
# Feature importance helper (shared by all tree-based models)
# ---------------------------------------------------------------------------

def get_feature_importance(model, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances.
    For LinearRegression, returns coefficients instead.
    Pass .best_estimator_ if the model is a CV search object.
    """
    estimator = getattr(model, "best_estimator_", model)

    if isinstance(estimator, LinearRegression):
        importances = estimator.coef_
    else:
        importances = estimator.feature_importances_

    df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances.astype(float)})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)



def get_feature_importance_decile(model: dict, X_train: dict) -> pd.DataFrame:
    all_importances = []

    for key, fitted_model in model.items():
        importance_df = pd.DataFrame({
            'Feature': X_train[key].columns,
            'Importance': fitted_model.feature_importances_.astype(float),
            'bin': key,
        }).sort_values('Importance', ascending=False)
        all_importances.append(importance_df)

    final_importance_df = pd.concat(all_importances, ignore_index=True)

    bin_rename = {
        bin_name: f'Importance_{i+1}'
        for i, bin_name in enumerate(final_importance_df['bin'].unique())
    }

    return (
        final_importance_df
        .pivot_table(index='Feature', columns='bin', values='Importance')
        .rename(columns=bin_rename)
        .reset_index()
    )

    

# ---------------------------------------------------------------------------
# 1. Linear Regression
# ---------------------------------------------------------------------------

def train_linear_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> LinearRegression:
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 2. Random Forest
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 500,
    max_depth: int = 25,
    min_samples_split: int = 20,
    min_samples_leaf: int = 2,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 3. XGBoost Simple (quick benchmark, no tuning)
# ---------------------------------------------------------------------------

def train_xgb_simple(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 25,
) -> XGBRegressor:
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=n_estimators,
        seed=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# 4. XGBoost Tuned (RandomizedSearchCV)
# ---------------------------------------------------------------------------

def train_xgb_tuned(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv: int = 5,
):
    """
    XGBoost regressor with RandomizedSearchCV.
    Returns the fitted GridSearchCV object; call .best_estimator_ for the model.
    """
    base_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.0,
        random_state=RANDOM_STATE,
    )

    param_dist = {
        "n_estimators": [500, 1000, 1500],
        "max_depth": [3, 4, 5],
        "min_child_weight": [20, 40, 60],
        "subsample": [0.5, 0.6, 0.7],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0.7, 0.9, 0.95],
        "reg_lambda": [2, 5, 7],
        "learning_rate": [0.01, 0.02],
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_mean_absolute_error",
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        cv=cv,
    )
    search.fit(X_train, y_train)

    print("Best params:", search.best_params_)
    print(f"Best CV MAE: ${-search.best_score_:,.2f}")
    return search


# ---------------------------------------------------------------------------
# 5. Class-based XGBoost (one model per CLV tier)
# ---------------------------------------------------------------------------

def train_xgb_by_rank(
    X_train_dict: dict,
    y_train_dict: dict,
    n_iter: int = 30,
    cv: int = 5,
) -> dict:
    """
    Train one tuned XGBoost regressor per CLV tier (low / mid / high).

    Parameters
    ----------
    X_train_dict : {'low': df, 'mid': df, 'high': df}  — from model_prep
    y_train_dict : {'low': series, 'mid': series, 'high': series}

    Returns
    -------
    dict of fitted RandomizedSearchCV objects keyed by rank name.
    """
    param_dist = {
        "max_depth": [3, 4, 5],
        "min_child_weight": [5, 20, 30],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.7, 1.0],
        "reg_alpha": [0, 0.1, 0.5],
        "reg_lambda": [1, 2, 5],
        "learning_rate": [0.02, 0.05],
    }

    searches = {}
    for rank in RANKING_LABELS:
        if rank not in X_train_dict:
            continue

        base = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=10,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            gamma=0.0,
            random_state=RANDOM_STATE,
        )

        search = RandomizedSearchCV(
            estimator=base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            verbose=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            cv=cv,
        )
        search.fit(X_train_dict[rank], y_train_dict[rank])

        print(f"\n[{rank}] Best params: {search.best_params_}")
        print(f"[{rank}] Best CV MAE: ${-search.best_score_:,.2f}")
        searches[rank] = search

    return searches


def predict_by_rank(
    rank_models: dict,
    X_test_dict: dict,
) -> dict:
    """
    Run predictions for each rank using the corresponding model.
    Returns {'low': array, 'mid': array, 'high': array}.
    """
    return {
        rank: rank_models[rank].predict(X_test_dict[rank])
        for rank in rank_models
        if rank in X_test_dict
    }
