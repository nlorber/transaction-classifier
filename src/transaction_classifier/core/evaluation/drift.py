"""Population Stability Index (PSI) drift detection.

Compares a production batch against a reference distribution frozen at training
time. Two families are scored:

* **input drift** — low-cardinality columns derived from the raw transaction
  frame. The assembled feature matrix is deliberately *not* used: it is ~2000
  sparse TF-IDF dimensions, where a per-column PSI carries no statistical
  meaning.
* **output drift** — the predicted class distribution and the confidence
  histogram, neither of which needs ground-truth labels.

``bin_edges`` always holds *interior* edges (``n_bins - 1`` of them), the form
``np.digitize`` expects. Values outside the training range fall into the first
or last bin, so no clipping is required.
"""

from typing import Any

import numpy as np
import pandas as pd

from ..features.standard import build_date_features, build_numeric_features

# Standard scorecard-monitoring thresholds.
PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25

# Guards the log-ratio against empty bins, which would otherwise diverge.
_EPSILON = 1e-4

_N_BINS = 10

# Continuous features need quantile edges computed from the training data.
_CONTINUOUS_FEATURES: tuple[str, ...] = ("amount", "desc_len")

# Categorical features use fixed category sets that are identical at training
# and inference by construction — _AMOUNT_EDGES is a module constant rather than
# learnt bins, and weekday/binary flags are structurally bounded. Nothing to
# serialise, and no "unseen category" case to handle.
_CATEGORICAL_FEATURES: dict[str, list[int]] = {
    "is_debit": [0, 1],
    "has_reference": [0, 1],
    "amount_bucket": list(range(9)),  # _AMOUNT_EDGES holds 10 edges -> 9 buckets
    "weekday": list(range(7)),
}

# Confidence is a probability, so its bins are fixed rather than data-derived.
_CONFIDENCE_EDGES: list[float] = [round(0.1 * i, 1) for i in range(1, _N_BINS)]


def compute_psi(
    ref_proportions: list[float],
    prod_proportions: list[float],
    epsilon: float = _EPSILON,
) -> float:
    """Population Stability Index between two aligned proportion vectors."""
    ref = np.clip(np.asarray(ref_proportions, dtype=float), epsilon, None)
    prod = np.clip(np.asarray(prod_proportions, dtype=float), epsilon, None)
    return float(np.sum((prod - ref) * np.log(prod / ref)))


def psi_verdict(psi: float) -> str:
    """Map a PSI value onto ``stable`` / ``moderate`` / ``significant``."""
    if psi < PSI_STABLE_MAX:
        return "stable"
    if psi < PSI_MODERATE_MAX:
        return "moderate"
    return "significant"


def build_baseline(
    train_df: pd.DataFrame,
    val_proba: np.ndarray[Any, np.dtype[Any]],
    classes: np.ndarray[Any, np.dtype[Any]],
    n_bins: int = _N_BINS,
) -> dict[str, Any]:
    """Compute the drift reference from the training split.

    ``val_proba`` is ``predict_proba`` on the *held-out validation* set — a model
    is always over-confident on its own training rows, so validation gives a
    reference that matches production behaviour.

    Output references are the model's own validation predictions, not the true
    label distribution: an imbalanced multi-class model systematically
    under-predicts rare classes, so comparing production predictions against
    ground-truth proportions would measure that bias rather than drift.

    The two reference families therefore rest on different row counts, and both
    are recorded: ``reference_size`` is the training rows behind the input
    references, ``output_reference_size`` the validation rows behind the output
    ones. Callers judging whether a score is trustworthy need the matching one.

    Returns a JSON-serialisable dict for ``Manifest.drift_baseline``.
    """
    features = _feature_frame(train_df)

    input_features: dict[str, Any] = {}
    for name in _CONTINUOUS_FEATURES:
        values = features[name].to_numpy(dtype=float)
        edges = _quantile_edges(values, n_bins)
        input_features[name] = {
            "kind": "continuous",
            "bin_edges": edges,
            "ref_proportions": _continuous_proportions(values, edges),
        }
    for name, categories in _CATEGORICAL_FEATURES.items():
        input_features[name] = {
            "kind": "categorical",
            "categories": categories,
            "ref_proportions": _categorical_proportions(features[name], categories),
        }

    class_names = [str(c) for c in classes]
    predicted = np.asarray(class_names)[np.argmax(val_proba, axis=1)]
    max_confidence = np.max(val_proba, axis=1)

    return {
        "schema_version": 1,
        "reference_size": int(len(train_df)),
        "output_reference_size": int(len(val_proba)),
        "input_features": input_features,
        "output": {
            "predicted_class_distribution": {
                "categories": class_names,
                "ref_proportions": _categorical_proportions(predicted, class_names),
            },
            "confidence": {
                "bin_edges": _CONFIDENCE_EDGES,
                "ref_proportions": _continuous_proportions(max_confidence, _CONFIDENCE_EDGES),
            },
        },
    }


def evaluate_drift(
    frame: pd.DataFrame,
    proba: np.ndarray[Any, np.dtype[Any]],
    classes: np.ndarray[Any, np.dtype[Any]],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    """Score a production batch against a frozen baseline.

    ``frame`` matches ``Predictor.build_frame`` output; ``classes`` is
    ``label_encoder.classes_``, giving the column order of ``proba``.

    An input feature with no usable rows in *frame* is omitted from
    ``input_drift``, so callers must treat that mapping's keys as a subset of
    the baseline's rather than a fixed set.
    """
    features = _feature_frame(frame)

    input_drift: dict[str, Any] = {}
    for name, spec in baseline["input_features"].items():
        if spec["kind"] == "continuous":
            observed = _continuous_proportions(
                features[name].to_numpy(dtype=float), spec["bin_edges"]
            )
        else:
            observed = _categorical_proportions(features[name], spec["categories"])
        if not any(observed):
            # Every row was excluded from this feature (undated rows for
            # weekday), leaving nothing to compare. Omitting the entry says so;
            # scoring an all-empty vector would report a maximal PSI instead.
            continue
        input_drift[name] = _score(spec["ref_proportions"], observed)

    predicted_spec = baseline["output"]["predicted_class_distribution"]
    confidence_spec = baseline["output"]["confidence"]
    predicted = np.asarray([str(c) for c in classes])[np.argmax(proba, axis=1)]
    max_confidence = np.max(proba, axis=1)

    output_drift = {
        "predicted_class_distribution": _score(
            predicted_spec["ref_proportions"],
            _categorical_proportions(predicted, predicted_spec["categories"]),
        ),
        "confidence_distribution": _score(
            confidence_spec["ref_proportions"],
            _continuous_proportions(max_confidence, confidence_spec["bin_edges"]),
        ),
    }

    worst = max(entry["psi"] for entry in (*input_drift.values(), *output_drift.values()))
    return {
        "input_drift": input_drift,
        "output_drift": output_drift,
        "overall_verdict": psi_verdict(worst),
    }


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the drift-tracked columns from a raw transaction frame.

    Rows whose ``posting_date`` does not parse are excluded from ``weekday``:
    ``build_date_features`` imputes those to a single weekday, and scoring the
    imputed value would read an absent optional field as a hard shift in the
    weekday distribution. Excluded rows drop out of both the numerator and the
    denominator, so the remaining proportions still sum to 1.
    """
    features = pd.concat([build_numeric_features(frame), build_date_features(frame)], axis=1)
    dated = pd.to_datetime(frame["posting_date"], errors="coerce").notna()
    features["weekday"] = features["weekday"].astype(float).where(dated)
    return features


def _quantile_edges(values: np.ndarray[Any, np.dtype[Any]], n_bins: int) -> list[float]:
    """Interior quantile edges. Duplicates on tiny samples are harmless."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    return [float(edge) for edge in np.quantile(values, quantiles)]


def _continuous_proportions(
    values: np.ndarray[Any, np.dtype[Any]], edges: list[float]
) -> list[float]:
    counts = np.bincount(np.digitize(values, edges), minlength=len(edges) + 1)
    return _normalise(counts.astype(float))


def _categorical_proportions(values: Any, categories: list[Any]) -> list[float]:
    counts = pd.Series(values).value_counts()
    return _normalise(np.array([float(counts.get(category, 0)) for category in categories]))


def _normalise(counts: np.ndarray[Any, np.dtype[Any]]) -> list[float]:
    total = float(counts.sum())
    if total == 0.0:
        return [0.0] * len(counts)
    return [float(count / total) for count in counts]


def _score(ref_proportions: list[float], prod_proportions: list[float]) -> dict[str, Any]:
    """Round before judging so the reported PSI and verdict never disagree."""
    psi = round(compute_psi(ref_proportions, prod_proportions), 4)
    return {"psi": psi, "verdict": psi_verdict(psi)}
