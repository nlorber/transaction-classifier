"""Versioned model storage with atomic promotion and hot-reload detection."""

import hashlib
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.preprocessing import LabelEncoder

from .schema import Manifest, ModelBundle

logger = logging.getLogger(__name__)


def _file_sha256(filepath: Path) -> str:
    """Compute the SHA-256 digest of a file in streaming fashion."""
    digest = hashlib.sha256()
    with open(filepath, "rb") as fh:
        while chunk := fh.read(8192):
            digest.update(chunk)
    return digest.hexdigest()


class ModelStore:
    """Manages versioned model bundles on disk.

    Directory layout::

        root/
          v-20260301-120000/
            classifier.json
            classifier.meta.json   (auto-created by model.persist)
            text_features.joblib
            label_encoder.joblib
            manifest.json
          current -> v-20260301-120000   (symlink)
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._active_target: str | None = None

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(
        self,
        model: Any,
        text_extractor: Any,
        label_encoder: LabelEncoder,
        metrics: dict[str, Any],
        config: dict[str, Any],
        n_features: int = 0,
    ) -> Manifest:
        """Persist a complete model bundle and return its manifest."""
        tag = f"v-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
        dest = self.root / tag
        dest.mkdir(parents=True, exist_ok=True)

        # Model (also creates classifier.meta.json automatically)
        model_file = dest / "classifier.json"
        model.persist(model_file)

        # Text features
        vec_file = dest / "text_features.joblib"
        joblib.dump(text_extractor.vectorizer_dict, vec_file)

        # Label encoder
        enc_file = dest / "label_encoder.joblib"
        joblib.dump(label_encoder, enc_file)

        # Checksums (skip classifier.meta.json — auto-generated sidecar)
        checksums = {
            "classifier.json": _file_sha256(model_file),
            "text_features.joblib": _file_sha256(vec_file),
            "label_encoder.joblib": _file_sha256(enc_file),
        }

        manifest = Manifest(
            version=tag,
            config=config,
            metrics=metrics,
            num_categories=model.n_classes_ or 0,
            n_features=n_features,
            checksums=checksums,
            status="candidate",
        )
        (dest / "manifest.json").write_text(manifest.model_dump_json(indent=2))

        logger.info("Stored bundle %s in %s", tag, dest)
        return manifest

    # ------------------------------------------------------------------
    # Promote
    # ------------------------------------------------------------------

    def promote(self, version: str) -> None:
        """Atomically switch the *current* symlink to the given version."""
        target = self.root / version
        if not target.exists():
            raise FileNotFoundError(f"Version directory not found: {target}")

        link = self.root / "current"
        tmp = self.root / f".current_swap_{os.getpid()}"

        if tmp.exists() or tmp.is_symlink():
            tmp.unlink()
        tmp.symlink_to(target.resolve())
        tmp.rename(link)

        logger.info("Promoted %s → current", version)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load_active(self) -> ModelBundle:
        """Load the bundle that the *current* symlink points to."""
        link = self.root / "current"
        if not link.exists():
            raise FileNotFoundError(
                f"No 'current' symlink in {self.root}. Run training with --auto-promote first."
            )

        target = self._resolve_current_target()
        self._active_target = target.name
        return self._load_from(target)

    def has_update(self) -> bool:
        """Return *True* when the symlink has moved since the last load."""
        link = self.root / "current"
        if not link.is_symlink():
            return False
        return self._resolve_current_target().name != self._active_target

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def available_versions(self) -> list[str]:
        """List all version directories, newest first."""
        return sorted(
            (d.name for d in self.root.iterdir() if d.is_dir() and d.name.startswith("v-")),
            reverse=True,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _resolve_current_target(self) -> Path:
        """Resolve the *current* symlink to an absolute version directory."""
        link = self.root / "current"
        target = Path(os.readlink(link))
        if not target.is_absolute():
            target = (self.root / target).resolve()
        return target

    def _verify_checksums(self, version_dir: Path, manifest: Manifest) -> None:
        for fname, expected in manifest.checksums.items():
            fp = version_dir / fname
            if not fp.exists():
                raise FileNotFoundError(f"Missing artefact: {fp}")
            actual = hashlib.sha256(fp.read_bytes()).hexdigest()
            if actual != expected:
                raise RuntimeError(
                    f"Hash mismatch for {fname}: expected {expected[:16]}…, got {actual[:16]}…"
                )
        logger.info("All hashes verified for %s", manifest.version)

    def _load_from(self, version_dir: Path) -> ModelBundle:
        logger.info("Loading bundle from %s", version_dir)

        manifest = Manifest.model_validate_json((version_dir / "manifest.json").read_text())
        if manifest.checksums:
            self._verify_checksums(version_dir, manifest)

        # Runtime imports to avoid circular dependencies
        from ..features.text import TfidfFeatureExtractor
        from ..models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.restore(version_dir / "classifier.json")

        vecs = joblib.load(version_dir / "text_features.joblib")
        extractor = TfidfFeatureExtractor()
        extractor.vec_label = vecs["label"]
        extractor.vec_detail = vecs["detail"]
        extractor.vec_char = vecs["char"]
        extractor._fitted = True

        encoder = joblib.load(version_dir / "label_encoder.joblib")

        logger.info(
            "Loaded %s (%d categories, %d features)",
            manifest.version,
            manifest.num_categories,
            manifest.n_features,
        )
        return ModelBundle(
            model=model,
            text_extractor=extractor,
            label_encoder=encoder,
            manifest=manifest,
        )
