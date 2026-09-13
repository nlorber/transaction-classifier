"""Tests for the XGBoost model wrapper's inference path."""

import numpy as np
import pytest
import xgboost as xgb
from sklearn.datasets import make_classification

from transaction_classifier.core.models.xgboost_model import XGBoostModel


@pytest.fixture(scope="module")
def early_stopped():
    """A model whose early stopping leaves trailing trees past best_iteration."""
    features, labels = make_classification(
        n_samples=1500,
        n_features=20,
        n_informative=8,
        n_classes=4,
        flip_y=0.3,
        random_state=0,
    )
    model = XGBoostModel(n_estimators=400, learning_rate=0.3, patience=10, verbosity=0)
    model.fit(features[:1000], labels[:1000], X_val=features[1000:], y_val=labels[1000:])
    assert model.model.best_iteration + 1 < model.model.get_booster().num_boosted_rounds()
    return model, features[1000:]


def _best_iteration_proba(model, features):
    booster = model.model.get_booster()
    return booster.predict(
        xgb.DMatrix(features), iteration_range=(0, model.model.best_iteration + 1)
    )


def test_predict_proba_stops_at_best_iteration(early_stopped):
    model, features = early_stopped
    np.testing.assert_allclose(
        model.predict_proba(features), _best_iteration_proba(model, features), rtol=1e-6
    )


def test_restored_model_stops_at_best_iteration(early_stopped, tmp_path):
    model, features = early_stopped
    model.persist(tmp_path / "classifier.json")
    restored = XGBoostModel().restore(tmp_path / "classifier.json")
    np.testing.assert_allclose(
        restored.predict_proba(features), _best_iteration_proba(model, features), rtol=1e-6
    )


def test_predict_proba_without_early_stopping_uses_all_trees():
    features, labels = make_classification(
        n_samples=200, n_features=10, n_informative=5, n_classes=3, random_state=0
    )
    model = XGBoostModel(n_estimators=15, patience=None, verbosity=0).fit(features, labels)
    all_trees = model.model.get_booster().predict(xgb.DMatrix(features))
    np.testing.assert_allclose(model.predict_proba(features), all_trees, rtol=1e-6)
