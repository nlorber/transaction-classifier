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

    def check(
        self,
        manifest: Manifest,
        baseline_accuracy: float,
        n_classes: int,
    ) -> GateResult:
        """Evaluate the manifest's metrics against baseline + lift."""
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
