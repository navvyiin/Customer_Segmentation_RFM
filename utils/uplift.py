import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from econml.dr import DRLearner


def prepare_uplift_data(df, treatment_col, outcome_col, feature_cols):
    """
    Prepare dataset for uplift modelling.

    df: pandas DataFrame with historical campaign info
    treatment_col: 1 if customer received offer, else 0
    outcome_col: numeric revenue or conversion
    feature_cols: list of feature names
    """
    X = df[feature_cols]
    T = df[treatment_col]
    Y = df[outcome_col]
    return X, T, Y


def train_drl_uplift_model(X, T, Y):
    """
    Train a Double Robust Learner uplift model.
    This replaces CausalForestDML for full Windows compatibility.
    """

    model_y = RandomForestRegressor(n_estimators=200)
    model_t = RandomForestClassifier(n_estimators=200)

    model = DRLearner(
        model_regression=model_y,
        model_propensity=model_t
    )

    model.fit(Y, T, X=X)
    return model


def predict_uplift(model, X_new):
    """
    Predict individual treatment effects (uplift) for customers.
    """
    uplift = model.effect(X_new)
    return uplift


def uplift_table(df, uplift_scores, customer_id_col="CustomerID"):
    """
    Produce a simple table of customers with uplift score.
    """
    res = df[[customer_id_col]].copy()
    res["uplift"] = uplift_scores
    res = res.sort_values("uplift", ascending=False)
    return res