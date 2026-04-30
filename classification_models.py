"""
classification_models.py
------------------------
Trains and evaluates three classification models that predict CLV tier
(low / mid / high):
  1. Logistic Regression (with optional GridSearchCV tuning)
  2. Decision Tree (RandomizedSearchCV)
  3. XGBoost Classifier (RandomizedSearchCV)

Each train function returns a fitted model object.
Call evaluation.evaluate_classifier() to print metrics.
"""

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint
from xgboost import XGBClassifier

from src.config import RANDOM_STATE


# ---------------------------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------------------------

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    max_iter: int = 1000,
):
    """Baseline logistic regression (no hyperparameter search)."""
    model = LogisticRegression(max_iter=max_iter, random_state=RANDOM_STATE)
    model.fit(X_train, y_train_encoded)
    return model


def train_logistic_cv(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    cv: int = 5,
):
    """
    Logistic regression with L1/L2 penalty search via GridSearchCV.
    Uses a StandardScaler + LogisticRegression pipeline (saga solver).
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000, solver="saga", random_state=RANDOM_STATE)),
    ])

    param_grid = {
        "logreg__C": np.logspace(-5, 8, 15),
        "logreg__penalty": ["l1", "l2"],
    }

    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train_encoded)

    print("Best params:", grid.best_params_)
    print("Best CV accuracy:", grid.best_score_)
    return grid


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

def train_decision_tree(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    n_iter: int = 50,
    cv: int = 5,
):
    """
    Decision tree with RandomizedSearchCV over depth, features, and leaf size.
    """
    param_dist = {
        "max_depth": [3, 5, 7, None],
        "max_features": randint(1, X_train.shape[1]),
        "min_samples_leaf": randint(1, 9),
        "criterion": ["gini", "entropy"],
    }

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        tree, param_dist, n_iter=n_iter, cv=cv,
        scoring="f1_macro", n_jobs=-1, random_state=RANDOM_STATE,
    )
    search.fit(X_train, y_train_encoded)

    print("Best params:", search.best_params_)
    print("Best CV f1_macro:", search.best_score_)
    return search


# ---------------------------------------------------------------------------
# XGBoost Classifier
# ---------------------------------------------------------------------------

def train_xgb_classifier(
    X_train: pd.DataFrame,
    y_train_encoded: np.ndarray,
    n_iter: int = 25,
    cv: int = 5,
):
    """
    XGBoost multi-class classifier with RandomizedSearchCV.
    objective='multi:softprob' returns per-class probabilities required for
    f1_macro scoring.
    """
    base_model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=4,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=2.0,
        gamma=0.0,
        random_state=RANDOM_STATE,
        eval_metric="mlogloss",
    )

    param_dist = {
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
        scoring="f1_macro",
        verbose=1,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        cv=cv,
    )
    search.fit(X_train, y_train_encoded)

    print("Best params:", search.best_params_)
    print("Best CV f1_macro:", search.best_score_)
    return search


# ---------------------------------------------------------------------------
# Feature importance helper for tree-based classifiers
# ---------------------------------------------------------------------------

def get_classifier_feature_importance(
    model, X_train: pd.DataFrame
) -> pd.DataFrame:
    """
    Return a sorted DataFrame of feature importances.
    Works with sklearn trees and XGBoost estimators.
    Pass the .best_estimator_ if the model is a CV search object.
    """
    estimator = getattr(model, "best_estimator_", model)
    importances = estimator.feature_importances_
    df = pd.DataFrame({"Feature": X_train.columns, "Importance": importances.astype(float)})
    return df.sort_values("Importance", ascending=False).reset_index(drop=True)
