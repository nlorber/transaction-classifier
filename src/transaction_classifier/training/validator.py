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
    accuracy_floor: float
    balanced_accuracy_floor: float
    live_accuracy: float | None


class QualityGate:
    """Decides whether a trained model may replace the live one.

    A candidate must clear two independent floors:

    1. ``accuracy >= max(majority * (1 + min_lift), live - max_accuracy_drop)``
       -- it beats the majority-class baseline by a margin, and it is no worse
       than the promoted model beyond a small tolerance (non-inferiority).
       Without a promoted model only the majority term applies.
    2. ``balanced_accuracy >= 1/n_classes * (1 + min_lift)`` -- it beats chance
       on the macro view, so a majority-class predictor cannot pass on (1).
    """

    def __init__(self, min_lift: float = 0.20, max_accuracy_drop: float = 0.01):
        self.min_lift = min_lift
        self.max_accuracy_drop = max_accuracy_drop

    def _live_accuracy(self, store: ModelStore) -> float | None:
        """Return the promoted model's accuracy, or *None* if it cannot be read."""
        link = store.root / "current"
        if not link.exists():
            return None
        try:
            manifest_path = link / "manifest.json"
            live_manifest = Manifest.model_validate_json(manifest_path.read_text())
            return float(live_manifest.metrics["accuracy"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read live manifest, skipping non-inferiority: %s", exc)
            return None

    def check(
        self,
        manifest: Manifest,
        baseline_accuracy: float,
        n_classes: int,
        store: ModelStore | None = None,
    ) -> GateResult:
        """Evaluate the manifest's metrics against the gate floors.

        *baseline_accuracy* is the majority-class frequency of the training
        set. When *store* holds a promoted model, its accuracy
        (``models/current/manifest.json``) adds the non-inferiority floor.
        """
        # ponytail: the live accuracy was measured on its own evaluation window,
        # not the candidate's; max_accuracy_drop absorbs window-to-window noise.
        # Re-score the live bundle on the candidate's test set if that noise
        # ever exceeds the tolerance.
        live_acc = self._live_accuracy(store) if store is not None else None

        acc = manifest.metrics.get("accuracy", 0.0)
        bal = manifest.metrics.get("balanced_accuracy", 0.0)

        acc_floor = baseline_accuracy * (1 + self.min_lift)
        if live_acc is not None:
            acc_floor = max(acc_floor, live_acc - self.max_accuracy_drop)
        bal_floor = (1.0 / max(n_classes, 1)) * (1 + self.min_lift)

        ok = acc >= acc_floor and bal >= bal_floor

        log = logger.info if ok else logger.warning
        log(
            "Model %s %s gate (acc=%.4f, floor %.4f; bal=%.4f, floor %.4f; live=%s)",
            manifest.version,
            "passed" if ok else "FAILED",
            acc,
            acc_floor,
            bal,
            bal_floor,
            "none" if live_acc is None else f"{live_acc:.4f}",
        )

        return GateResult(
            passed=ok,
            accuracy=acc,
            balanced_accuracy=bal,
            accuracy_floor=acc_floor,
            balanced_accuracy_floor=bal_floor,
            live_accuracy=live_acc,
        )

    def approve_and_promote(self, vault: ModelStore, manifest: Manifest) -> None:
        """Promote the model to *current* in the vault."""
        vault.promote(manifest.version)
        logger.info("Promoted %s -> current", manifest.version)
