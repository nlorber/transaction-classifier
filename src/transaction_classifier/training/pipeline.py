"""End-to-end model training orchestration."""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.metrics import recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from ..core.artifacts.schema import Manifest
from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.data.source import DataSource
from ..core.data.splitter import split_by_date, temporal_partition_stats
from ..core.evaluation.drift import build_baseline
from ..core.evaluation.metrics import evaluate_predictions
from ..core.features.engine import DomainFeatureEngine
from ..core.features.pipeline import assemble_feature_matrix
from ..core.features.text import TfidfFeatureExtractor
from ..core.models.xgboost_model import XGBoostModel
from ..core.utils.reproducibility import get_reproducibility_info, set_seed

logger = logging.getLogger(__name__)

Labels = np.ndarray[Any, np.dtype[Any]]


class TrainingPipeline:
    """Coordinates data loading, feature creation, training, evaluation, and storage.

    Three chronological blocks keep evaluation honest: the model fits on
    *train* and early-stops on *val*, and every reported metric (and therefore
    the quality gate) comes from *test*, which no fitting decision has seen.
    """

    def __init__(self, settings: Settings, provider: DataSource):
        self.settings = settings
        self.provider = provider

    # ------------------------------------------------------------------

    def execute(self) -> tuple[Manifest, float, int]:
        """Run the full pipeline and return (manifest, baseline_accuracy, n_classes)."""
        cfg = self.settings
        set_seed(cfg.random_state)

        df = self._ingest()
        train_df, val_df, test_df, stats = self._split(df)
        le, y_train, baseline = self._encode_train_labels(train_df)
        val_known, y_val = self._known_classes(le, val_df, "val")
        test_known, y_test = self._known_classes(le, test_df, "test")
        extractor, _, X_train, X_val, X_test = self._build_features(
            cfg, train_df, val_known, test_known
        )
        model, train_secs = self._train(cfg, X_train, y_train, X_val, y_val)
        metrics = self._evaluate(model, X_test, y_test, le)
        drift_baseline = self._drift_baseline(train_df, model, X_test, le)
        manifest = self._persist(
            model,
            extractor,
            le,
            metrics,
            stats,
            len(val_known),
            len(test_known),
            X_train,
            train_secs,
            drift_baseline,
        )
        return manifest, baseline, len(le.classes_)

    # ------------------------------------------------------------------
    # Private pipeline steps
    # ------------------------------------------------------------------

    def _ingest(self) -> pd.DataFrame:
        cfg = self.settings
        logger.info("Fetching data from %s …", cfg.data_path)
        t0 = time.time()
        df = self.provider.fetch(
            min_class_samples=cfg.min_class_samples,
            target_length=cfg.target_length,
        )
        logger.info(
            "Fetched %d rows, %d classes (%.1fs)",
            len(df),
            df["target"].nunique(),
            time.time() - t0,
        )
        return df

    def _split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        cfg = self.settings
        logger.info("Splitting (temporal, train=%.2f, val=%.2f) …", cfg.train_ratio, cfg.val_ratio)
        train_df, val_df, test_df = split_by_date(
            df, train_ratio=cfg.train_ratio, val_ratio=cfg.val_ratio
        )
        stats = temporal_partition_stats(train_df, val_df, test_df)
        logger.info(
            "Train: %d | Val: %d | Test: %d | Classes: %d",
            stats["train_rows"],
            stats["val_rows"],
            stats["test_rows"],
            stats["train_n_classes"],
        )
        return train_df, val_df, test_df, stats

    def _encode_train_labels(self, train_df: pd.DataFrame) -> tuple[LabelEncoder, Labels, float]:
        le = LabelEncoder()
        y_train: Labels = le.fit_transform(train_df["target"])
        baseline = float(pd.Series(y_train).value_counts(normalize=True).iloc[0])
        return le, y_train, baseline

    @staticmethod
    def _known_classes(
        le: LabelEncoder, df: pd.DataFrame, name: str
    ) -> tuple[pd.DataFrame, Labels]:
        """Keep the rows whose class was seen in training; the model cannot score others."""
        known = df[df["target"].isin(le.classes_)]
        if len(known) < len(df):
            logger.warning("Excluded %d %s rows with unseen classes", len(df) - len(known), name)
        if known.empty:
            raise ValueError(f"The {name} block has no rows with a class seen in training")
        y: Labels = le.transform(known["target"])
        return known, y

    def _build_features(
        self,
        cfg: Settings,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> tuple[TfidfFeatureExtractor, DomainFeatureEngine, spmatrix, spmatrix, spmatrix]:
        logger.info("Building features …")
        t0 = time.time()
        extractor = TfidfFeatureExtractor(
            label_vocab_size=cfg.tfidf_max_label,
            detail_vocab_size=cfg.tfidf_max_detail,
            char_vocab_size=cfg.tfidf_max_char,
        )
        engine = DomainFeatureEngine(cfg.feature_profile)
        X_train = assemble_feature_matrix(train_df, extractor, engine, fit=True)  # noqa: N806
        X_val = assemble_feature_matrix(val_df, extractor, engine, fit=False)  # noqa: N806
        X_test = assemble_feature_matrix(test_df, extractor, engine, fit=False)  # noqa: N806
        logger.info("Feature shape: %s (%.1fs)", X_train.shape, time.time() - t0)
        return extractor, engine, X_train, X_val, X_test

    def _train(
        self,
        cfg: Settings,
        X_train: spmatrix,  # noqa: N803
        y_train: Labels,
        X_val: spmatrix,  # noqa: N803
        y_val: Labels,
    ) -> tuple[XGBoostModel, float]:
        logger.info(
            "Training XGBoostModel (%d rounds, depth=%d, balanced weights=%s) …",
            cfg.n_estimators,
            cfg.max_depth,
            cfg.balanced_class_weights,
        )
        t0 = time.time()
        model = XGBoostModel(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            patience=cfg.patience,
            max_bin=cfg.max_bin,
            random_state=cfg.random_state,
            device=cfg.device,
            log_every=10,
            checkpoint_dir=str(cfg.artifact_dir) + "/checkpoints",
        )
        # Early stopping still monitors the unweighted validation loss.
        weights = (
            compute_sample_weight("balanced", y_train) if cfg.balanced_class_weights else None
        )
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, sample_weight=weights)
        train_secs = time.time() - t0
        logger.info("Training finished (%.1fs)", train_secs)
        return model, train_secs

    def _evaluate(
        self,
        model: XGBoostModel,
        X_test: spmatrix,  # noqa: N803
        y_test: Labels,
        le: LabelEncoder,
    ) -> dict[str, Any]:
        logger.info("Evaluating on held-out test block …")
        y_hat = model.predict(X_test)
        report = evaluate_predictions(y_test, y_hat)
        logger.info(
            "Test accuracy=%.4f balanced=%.4f f1_weighted=%.4f",
            report.accuracy,
            report.balanced_accuracy,
            report.f1_weighted,
        )
        metrics: dict[str, Any] = report._asdict()
        # Averages hide whether rare codes are learned at all; record recall for
        # every class present in the test block (absent classes have no recall).
        present = np.unique(y_test)
        recall = recall_score(y_test, y_hat, labels=present, average=None, zero_division=0)
        metrics["per_class_recall"] = {
            str(code): round(float(value), 4)
            for code, value in zip(le.classes_[present], recall, strict=True)
        }
        return metrics

    def _drift_baseline(
        self,
        train_df: pd.DataFrame,
        model: XGBoostModel,
        X_test: spmatrix,  # noqa: N803
        le: LabelEncoder,
    ) -> dict[str, Any]:
        """Freeze the reference distributions used by /ops/drift."""
        logger.info("Computing drift baseline …")
        return build_baseline(train_df, model.predict_proba(X_test), le.classes_)

    def _persist(
        self,
        model: XGBoostModel,
        extractor: TfidfFeatureExtractor,
        le: LabelEncoder,
        metrics: dict[str, Any],
        stats: dict[str, Any],
        val_rows: int,
        test_rows: int,
        X_train: spmatrix,  # noqa: N803
        train_secs: float,
        drift_baseline: dict[str, Any],
    ) -> Manifest:
        cfg = self.settings
        logger.info("Storing artefacts …")
        store = ModelStore(cfg.artifact_dir)
        run_config = {
            "min_class_samples": cfg.min_class_samples,
            "target_length": cfg.target_length,
            "n_estimators": cfg.n_estimators,
            "max_depth": cfg.max_depth,
            "learning_rate": cfg.learning_rate,
            "patience": cfg.patience,
            "max_bin": cfg.max_bin,
            "balanced_class_weights": cfg.balanced_class_weights,
            "tfidf_max_label": cfg.tfidf_max_label,
            "tfidf_max_detail": cfg.tfidf_max_detail,
            "tfidf_max_char": cfg.tfidf_max_char,
            "random_state": cfg.random_state,
            "train_ratio": cfg.train_ratio,
            "val_ratio": cfg.val_ratio,
            "train_rows": stats["train_rows"],
            "val_rows": val_rows,
            "test_rows": test_rows,
            "num_categories": len(le.classes_),
            "train_seconds": train_secs,
            "env": get_reproducibility_info(),
        }
        manifest = store.save(
            model=model,
            text_extractor=extractor,
            label_encoder=le,
            metrics=metrics,
            config=run_config,
            n_features=X_train.shape[1],
            drift_baseline=drift_baseline,
        )
        logger.info("Stored version: %s", manifest.version)
        return manifest
