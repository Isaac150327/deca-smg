from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features import FEATURE_COLUMNS, ensure_feature_columns


class InsufficientTrainingDataError(ValueError):
    """Raised when a valid chronological model test cannot yet be formed."""


@dataclass
class WalkForwardResult:
    scored: pd.DataFrame
    final_model: Pipeline


def make_model(random_state: int = 42) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "regressor",
                HistGradientBoostingRegressor(
                    learning_rate=0.05,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=20,
                    l2_regularization=1.0,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _fit(frame: pd.DataFrame, random_state: int) -> Pipeline:
    ensure_feature_columns(frame)
    model = make_model(random_state)
    model.fit(frame[FEATURE_COLUMNS], frame["target_short_return_1d"])
    return model


def walk_forward_score(candidates: pd.DataFrame, config: dict) -> WalkForwardResult:
    """Generate strictly out-of-sample predictions and fit a final live model."""

    candidates = candidates.sort_values(["date", "ticker"]).copy()
    trainable = candidates.loc[candidates["target_short_return_1d"].notna()].copy()
    if trainable.empty:
        raise InsufficientTrainingDataError(
            "No candidates have a known next-snapshot return"
        )

    model_config = config["model"]
    dates = list(pd.Index(trainable["date"].drop_duplicates()).sort_values())
    minimum_dates = int(model_config["minimum_training_dates"])
    minimum_rows = int(model_config["minimum_training_rows"])
    rolling_dates = int(model_config["rolling_training_dates"])
    retrain_every = int(model_config["retrain_every_dates"])
    random_state = int(model_config["random_state"])

    predictions: list[pd.DataFrame] = []
    fitted_model: Pipeline | None = None
    for index, prediction_date in enumerate(dates):
        prior_dates = dates[max(0, index - rolling_dates) : index]
        if len(prior_dates) < minimum_dates:
            continue
        training = trainable.loc[trainable["date"].isin(prior_dates)]
        prediction_rows = trainable.loc[trainable["date"].eq(prediction_date)].copy()
        if len(training) < minimum_rows or prediction_rows.empty:
            continue
        if fitted_model is None or index % retrain_every == 0:
            fitted_model = _fit(training, random_state)
        prediction_rows["predicted_short_return"] = fitted_model.predict(
            prediction_rows[FEATURE_COLUMNS]
        )
        prediction_rows["training_rows"] = len(training)
        prediction_rows["training_end_date"] = max(prior_dates)
        predictions.append(prediction_rows)

    if not predictions:
        raise InsufficientTrainingDataError(
            "Not enough historical pump candidates for a walk-forward test. "
            f"Found {len(trainable):,} labeled candidates across {len(dates):,} dates; "
            f"the configured minimums are {minimum_rows:,} prior candidates across "
            f"{minimum_dates:,} prior dates."
        )

    final_model = _fit(trainable, random_state)
    return WalkForwardResult(pd.concat(predictions, ignore_index=True), final_model)


def score_live(model: Pipeline, candidates: pd.DataFrame) -> pd.DataFrame:
    ensure_feature_columns(candidates)
    result = candidates.copy()
    result["predicted_short_return"] = model.predict(result[FEATURE_COLUMNS])
    return result
