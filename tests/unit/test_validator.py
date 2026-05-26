"""Tests for the QualityGate post-training quality gate."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from transaction_classifier.core.artifacts.schema import Manifest
from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.training.validator import QualityGate


def _make_manifest(accuracy: float, balanced_accuracy: float) -> Manifest:
    return Manifest(
        version="v-test",
        metrics={"accuracy": accuracy, "balanced_accuracy": balanced_accuracy},
    )


class TestCheck:
    def test_quality_gate_passes_when_above_baseline(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is True

    def test_quality_gate_fails_when_below_baseline(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.35, balanced_accuracy=0.05)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_fails_when_accuracy_below_threshold(self):
        # baseline=0.40, min_lift=0.20 => acc_threshold=0.48
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.45, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_fails_when_balanced_accuracy_below_threshold(self):
        # n_classes=10 => bal_threshold=0.10*1.20=0.12
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.10)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_passes_at_exact_thresholds(self):
        # baseline=0.40, min_lift=0.20 => acc_threshold=0.48; n_classes=5 => bal_threshold=0.24
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.48, balanced_accuracy=0.24)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=5)
        assert result.passed is True

    def test_fails_when_both_below(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.10, balanced_accuracy=0.05)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_missing_metrics_default_to_zero_and_fail(self):
        gate = QualityGate(min_lift=0.20)
        manifest = Manifest(version="v-test", metrics={})
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_gate_result_fields_are_populated(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.accuracy == 0.60
        assert result.balanced_accuracy == 0.40
        assert result.baseline_accuracy == 0.40
        assert result.min_lift == 0.20


class TestApproveAndPromote:
    def test_calls_promote_on_vault(self):
        gate = QualityGate()
        vault = MagicMock()
        manifest = _make_manifest(accuracy=0.50, balanced_accuracy=0.25)
        gate.approve_and_promote(vault, manifest)
        vault.promote.assert_called_once_with("v-test")


class TestLiveBaseline:
    def _write_live_manifest(self, tmpdir: str, accuracy: float) -> None:
        """Write a minimal promoted manifest into a temp model store."""
        root = Path(tmpdir)
        version_dir = root / "v-live"
        version_dir.mkdir()
        manifest = Manifest(
            version="v-live",
            metrics={"accuracy": accuracy, "balanced_accuracy": 0.3},
        )
        (version_dir / "manifest.json").write_text(manifest.model_dump_json())
        link = root / "current"
        link.symlink_to(version_dir.resolve())

    def test_uses_live_model_accuracy_when_store_provided(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_live_manifest(tmpdir, accuracy=0.70)
            store = ModelStore(tmpdir)
            gate = QualityGate(min_lift=0.10)
            # Candidate beats live 0.70 by >10%
            manifest = _make_manifest(accuracy=0.80, balanced_accuracy=0.30)
            result = gate.check(manifest, baseline_accuracy=0.10, n_classes=5, store=store)
            assert result.baseline_accuracy == 0.70
            assert result.passed is True

    def test_falls_back_to_static_baseline_when_no_current_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ModelStore(tmpdir)  # no promoted model
            gate = QualityGate(min_lift=0.20)
            manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
            result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10, store=store)
            assert result.baseline_accuracy == 0.40
            assert result.passed is True

    def test_ignores_store_when_not_provided(self):
        """Without a store, behavior is unchanged."""
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.baseline_accuracy == 0.40

    def test_falls_back_on_corrupt_manifest(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            version_dir = root / "v-bad"
            version_dir.mkdir()
            (version_dir / "manifest.json").write_text("not valid json{{{")
            (root / "current").symlink_to(version_dir.resolve())
            store = ModelStore(tmpdir)
            gate = QualityGate(min_lift=0.20)
            manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
            result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10, store=store)
            assert result.baseline_accuracy == 0.40  # fallback
