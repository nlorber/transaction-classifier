"""Tests for the QualityGate post-training quality gate."""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transaction_classifier.core.artifacts.schema import Manifest
from transaction_classifier.core.artifacts.store import ModelStore
from transaction_classifier.training.validator import QualityGate


def _make_manifest(accuracy: float, balanced_accuracy: float, version: str = "v-test") -> Manifest:
    return Manifest(
        version=version,
        metrics={"accuracy": accuracy, "balanced_accuracy": balanced_accuracy},
    )


class TestCheckWithoutLiveModel:
    """Cold start: only the majority-class and chance floors apply."""

    def test_passes_when_above_both_floors(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is True

    def test_fails_when_accuracy_below_lift_floor(self):
        # baseline=0.40, min_lift=0.20 => floor=0.48
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.45, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_fails_when_balanced_accuracy_below_chance_floor(self):
        # n_classes=10 => floor=0.10*1.20=0.12
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.10)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_passes_at_exact_floors(self):
        # baseline=0.40 => acc floor 0.48; n_classes=5 => bal floor 0.24
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.48, balanced_accuracy=0.24)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=5)
        assert result.passed is True

    def test_missing_metrics_default_to_zero_and_fail(self):
        gate = QualityGate(min_lift=0.20)
        manifest = Manifest(version="v-test", metrics={})
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.passed is False

    def test_result_fields_are_populated(self):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(manifest, baseline_accuracy=0.40, n_classes=10)
        assert result.accuracy == 0.60
        assert result.balanced_accuracy == 0.40
        assert result.accuracy_floor == pytest.approx(0.48)
        assert result.balanced_accuracy_floor == pytest.approx(0.12)
        assert result.live_accuracy is None

    def test_empty_store_applies_lift_floor_only(self, tmp_path):
        gate = QualityGate(min_lift=0.20)
        manifest = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(
            manifest, baseline_accuracy=0.40, n_classes=10, store=ModelStore(tmp_path)
        )
        assert result.live_accuracy is None
        assert result.accuracy_floor == pytest.approx(0.48)
        assert result.passed is True


def _promote_live(root: Path, accuracy: float) -> ModelStore:
    """Write a promoted model manifest with the given accuracy into *root*."""
    version_dir = root / "v-live"
    version_dir.mkdir()
    live = _make_manifest(accuracy=accuracy, balanced_accuracy=0.45, version="v-live")
    (version_dir / "manifest.json").write_text(live.model_dump_json())
    store = ModelStore(root)
    store.promote("v-live")
    return store


class TestNonInferiorityAgainstLiveModel:
    """Retrain path: a candidate must not regress on the promoted model."""

    def test_equal_quality_candidate_is_promoted(self, tmp_path):
        store = _promote_live(tmp_path, accuracy=0.585)
        candidate = _make_manifest(accuracy=0.585, balanced_accuracy=0.49, version="v-new")
        (tmp_path / "v-new").mkdir()

        gate = QualityGate(min_lift=0.20, max_accuracy_drop=0.01)
        result = gate.check(candidate, baseline_accuracy=0.05, n_classes=80, store=store)
        assert result.passed is True
        assert result.live_accuracy == 0.585

        gate.approve_and_promote(store, candidate)
        assert Path(os.readlink(tmp_path / "current")).name == "v-new"

    def test_candidate_within_tolerance_passes(self, tmp_path):
        store = _promote_live(tmp_path, accuracy=0.585)
        gate = QualityGate(min_lift=0.20, max_accuracy_drop=0.01)
        candidate = _make_manifest(accuracy=0.578, balanced_accuracy=0.49)
        result = gate.check(candidate, baseline_accuracy=0.05, n_classes=80, store=store)
        assert result.accuracy_floor == pytest.approx(0.575)
        assert result.passed is True

    def test_clearly_worse_candidate_is_rejected(self, tmp_path):
        store = _promote_live(tmp_path, accuracy=0.585)
        gate = QualityGate(min_lift=0.20, max_accuracy_drop=0.01)
        candidate = _make_manifest(accuracy=0.55, balanced_accuracy=0.49)
        result = gate.check(candidate, baseline_accuracy=0.05, n_classes=80, store=store)
        assert result.passed is False

    def test_lift_floor_still_applies_when_live_model_is_weak(self, tmp_path):
        # Live 0.30 would allow 0.29, but the majority floor is 0.40*1.2=0.48.
        store = _promote_live(tmp_path, accuracy=0.30)
        gate = QualityGate(min_lift=0.20, max_accuracy_drop=0.01)
        candidate = _make_manifest(accuracy=0.35, balanced_accuracy=0.40)
        result = gate.check(candidate, baseline_accuracy=0.40, n_classes=10, store=store)
        assert result.accuracy_floor == pytest.approx(0.48)
        assert result.passed is False

    def test_corrupt_live_manifest_falls_back_to_lift_floor(self, tmp_path):
        version_dir = tmp_path / "v-bad"
        version_dir.mkdir()
        (version_dir / "manifest.json").write_text("not valid json{{{")
        (tmp_path / "current").symlink_to(version_dir.resolve())

        gate = QualityGate(min_lift=0.20)
        candidate = _make_manifest(accuracy=0.60, balanced_accuracy=0.40)
        result = gate.check(
            candidate, baseline_accuracy=0.40, n_classes=10, store=ModelStore(tmp_path)
        )
        assert result.live_accuracy is None
        assert result.passed is True


class TestApproveAndPromote:
    def test_calls_promote_on_vault(self):
        gate = QualityGate()
        vault = MagicMock()
        manifest = _make_manifest(accuracy=0.50, balanced_accuracy=0.25)
        gate.approve_and_promote(vault, manifest)
        vault.promote.assert_called_once_with("v-test")
