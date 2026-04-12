"""Unit tests for the Predictor."""

import tempfile

import pytest
from sklearn.preprocessing import LabelEncoder

from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.core.features.text import TfidfFeatureExtractor
from transaction_classifier.core.models.xgboost_model import XGBoostModel
from transaction_classifier.inference.predictor import Predictor
from transaction_classifier.inference.schemas import TransactionPayload


@pytest.fixture()
def domain_engine():
    from transaction_classifier.core.features.engine import DomainFeatureEngine

    return DomainFeatureEngine("config/profiles/french_treasury.yaml")


@pytest.fixture
def engine_with_model(sample_df, domain_engine):
    """Create a Predictor with a tiny trained model."""
    from transaction_classifier.core.features.pipeline import assemble_feature_matrix

    tf = TfidfFeatureExtractor(
        label_vocab_size=20, detail_vocab_size=20, char_vocab_size=20, min_df=1
    )
    X = assemble_feature_matrix(sample_df, tf, domain_engine, fit=True)
    le = LabelEncoder()
    y = le.fit_transform(sample_df["target"])
    model = XGBoostModel(n_estimators=5, max_depth=2, verbosity=0, patience=None)
    model.fit(X, y)

    with tempfile.TemporaryDirectory() as tmpdir:
        store = ModelStore(tmpdir)
        manifest = store.save(model, tf, le, {"accuracy": 0.5}, {})
        store.promote(manifest.version)

        bundle = store.load_active()
        yield Predictor(bundle, default_top_k=3, domain_engine=domain_engine)


def test_classify_single(engine_with_model):
    txs = [
        TransactionPayload(
            description="URSSAF COTISATIONS",
            remarks="PRLV SEPA CPY:FR123 NBE:URSSAF",
            debit=1234.56,
            posting_date="2025-01-15",
        )
    ]
    results = engine_with_model.classify(txs)
    assert len(results) == 1
    assert len(results[0].predictions) == 3
    assert all(p.confidence >= 0 for p in results[0].predictions)
    assert results[0].predictions[0].confidence >= results[0].predictions[1].confidence


def test_classify_batch(engine_with_model, sample_transactions):
    results = engine_with_model.classify(sample_transactions, top_k=2)
    assert len(results) == 2
    for r in results:
        assert len(r.predictions) == 2


def test_classify_custom_top_k(engine_with_model):
    txs = [TransactionPayload(description="TEST", posting_date="2025-01-01")]
    results = engine_with_model.classify(txs, top_k=1)
    assert len(results) == 1
    assert len(results[0].predictions) == 1


def test_explain_returns_contributions(engine_with_model):
    """Explain should return one result per transaction with feature contributions."""
    pytest.importorskip("shap")
    txns = [
        TransactionPayload(
            description="URSSAF COTISATIONS",
            remarks="PRLV SEPA CPY:FR123",
            debit=1234.56,
            posting_date="2025-01-15",
        ),
    ]
    results = engine_with_model.explain(txns, max_features=5)
    assert len(results) == 1
    item = results[0]
    assert item.predicted_code != ""
    assert 0 <= item.confidence <= 1
    assert len(item.contributions) <= 5
    for c in item.contributions:
        assert isinstance(c.feature, str)
        assert isinstance(c.shap_value, float)


def test_explain_max_features_limits_output(engine_with_model):
    """The max_features param should cap the number of contributions."""
    pytest.importorskip("shap")
    txns = [TransactionPayload(description="EDF FACTURE", debit=89.0)]
    results = engine_with_model.explain(txns, max_features=3)
    assert len(results[0].contributions) <= 3
