"""Post-training quality gate for model promotion."""

import logging
from dataclasses import dataclass

from ..core.artifacts.schema import Manifest
from ..core.artifacts.store import ModelStore

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Outcome of the quality-gate check."""

    passed: bool
    accuracy: float
    balanced_accuracy: float
    baseline_accuracy: float
    min_lift: float


class QualityGate:
    """Decides whether a trained model exceeds a baseline by a minimum margin.

    Rather than using fixed thresholds, the gate requires:
    1. accuracy > baseline_accuracy * (1 + min_lift)
    2. balanced_accuracy > 1/n_classes * (1 + min_lift)

    This adapts automatically to dataset difficulty and class count.
    """

    def __init__(self, min_lift: float = 0.20):
        self.min_lift = min_lift

    def _live_baseline(self, store: ModelStore, fallback: float) -> float:
        """Return the promoted model's accuracy, or *fallback* if none exists."""
        link = store.root / "current"
        if not link.exists():
            return fallback
        try:
            manifest_path = link / "manifest.json"
            live_manifest = Manifest.model_validate_json(manifest_path.read_text())
            live_acc = float(live_manifest.metrics.get("accuracy", 0.0))
            logger.info("Using live model accuracy as baseline: %.4f", live_acc)
            return live_acc
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not read live manifest, falling back to static baseline: %s", exc
            )
            return fallback

    def check(
        self,
        manifest: Manifest,
        baseline_accuracy: float,
        n_classes: int,
        store: ModelStore | None = None,
    ) -> GateResult:
        """Evaluate the manifest's metrics against baseline + lift.

        When *store* is provided the baseline is taken from the live model's
        accuracy (``models/current/manifest.json``); the static
        *baseline_accuracy* is used only as a fallback when no promoted model
        exists yet.
        """
        if store is not None:
            baseline_accuracy = self._live_baseline(store, fallback=baseline_accuracy)

        acc = manifest.metrics.get("accuracy", 0.0)
        bal = manifest.metrics.get("balanced_accuracy", 0.0)

        acc_threshold = baseline_accuracy * (1 + self.min_lift)
        bal_threshold = (1.0 / max(n_classes, 1)) * (1 + self.min_lift)

        ok = acc >= acc_threshold and bal >= bal_threshold

        if ok:
            logger.info(
                "Model %s passed gate (acc=%.4f >= %.4f, bal=%.4f >= %.4f)",
                manifest.version,
                acc,
                acc_threshold,
                bal,
                bal_threshold,
            )
        else:
            logger.warning(
                "Model %s FAILED gate (acc=%.4f < %.4f or bal=%.4f < %.4f)",
                manifest.version,
                acc,
                acc_threshold,
                bal,
                bal_threshold,
            )

        return GateResult(
            passed=ok,
            accuracy=acc,
            balanced_accuracy=bal,
            baseline_accuracy=baseline_accuracy,
            min_lift=self.min_lift,
        )

    def approve_and_promote(self, vault: ModelStore, manifest: Manifest) -> None:
        """Promote the model to *current* in the vault."""
        vault.promote(manifest.version)
        logger.info("Promoted %s -> current", manifest.version)
