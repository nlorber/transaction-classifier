"""End-to-end model training orchestration."""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.preprocessing import LabelEncoder

from ..core.artifacts.schema import Manifest
from ..core.artifacts.store import ModelStore
from ..core.config import Settings
from ..core.data.source import DataSource
from ..core.data.splitter import split_by_date, temporal_partition_stats
from ..core.evaluation.metrics import evaluate_predictions
from ..core.features.engine import DomainFeatureEngine
from ..core.features.pipeline import assemble_feature_matrix
from ..core.features.text import TfidfFeatureExtractor
from ..core.models.xgboost_model import XGBoostModel
from ..core.utils.reproducibility import get_reproducibility_info, set_seed

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Coordinates data loading, feature creation, training, evaluation, and storage."""

    def __init__(self, settings: Settings, provider: DataSource):
        self.settings = settings
        self.provider = provider

    # ------------------------------------------------------------------

    def execute(self) -> tuple[Manifest, float, int]:
        """Run the full pipeline and return (manifest, baseline_accuracy, n_classes)."""
        cfg = self.settings
        set_seed(cfg.random_state)

        df = self._ingest()
        train_df, val_df, stats = self._split(df)
        le, y_train, y_val, val_known, baseline = self._encode_labels(train_df, val_df)
        extractor, engine, X_train, X_val = self._build_features(cfg, train_df, val_known)
        model, train_secs = self._train(cfg, X_train, y_train, X_val, y_val)
        metrics = self._evaluate(model, X_val, y_val)
        manifest = self._persist(
            model, extractor, le, metrics, stats, len(val_known), X_train, train_secs
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

    def _split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        cfg = self.settings
        logger.info("Splitting (temporal, ratio=%.2f) …", cfg.train_ratio)
        train_df, val_df = split_by_date(df, train_ratio=cfg.train_ratio)
        stats = temporal_partition_stats(train_df, val_df)
        logger.info(
            "Train: %d | Val: %d | Classes: %d",
            stats["train_rows"],
            stats["val_rows"],
            stats["train_n_classes"],
        )
        return train_df, val_df, stats

    def _encode_labels(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame
    ) -> tuple[
        LabelEncoder,
        np.ndarray[Any, np.dtype[Any]],
        np.ndarray[Any, np.dtype[Any]],
        pd.DataFrame,
        float,
    ]:
        le = LabelEncoder()
        y_train: np.ndarray[Any, np.dtype[Any]] = le.fit_transform(train_df["target"])
        baseline = float(pd.Series(y_train).value_counts(normalize=True).iloc[0])
        known_mask = val_df["target"].isin(le.classes_)
        val_known = val_df[known_mask]
        y_val: np.ndarray[Any, np.dtype[Any]] = le.transform(val_known["target"])
        if len(val_known) < len(val_df):
            logger.warning(
                "Excluded %d val rows with unseen classes",
                len(val_df) - len(val_known),
            )
        return le, y_train, y_val, val_known, baseline

    def _build_features(
        self,
        cfg: Settings,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
    ) -> tuple[TfidfFeatureExtractor, DomainFeatureEngine, spmatrix, spmatrix]:
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
        logger.info("Feature shape: %s (%.1fs)", X_train.shape, time.time() - t0)
        return extractor, engine, X_train, X_val

    def _train(
        self,
        cfg: Settings,
        X_train: spmatrix,  # noqa: N803
        y_train: np.ndarray[Any, np.dtype[Any]],
        X_val: spmatrix,  # noqa: N803
        y_val: np.ndarray[Any, np.dtype[Any]],
    ) -> tuple[XGBoostModel, float]:
        logger.info(
            "Training XGBoostModel (%d rounds, depth=%d) …",
            cfg.n_estimators,
            cfg.max_depth,
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
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        train_secs = time.time() - t0
        logger.info("Training finished (%.1fs)", train_secs)
        return model, train_secs

    def _evaluate(
        self,
        model: XGBoostModel,
        X_val: spmatrix,  # noqa: N803
        y_val: np.ndarray[Any, np.dtype[Any]],
    ) -> dict[str, Any]:
        logger.info("Evaluating on validation set …")
        y_hat = model.predict(X_val)
        report = evaluate_predictions(y_val, y_hat)
        logger.info(
            "Val accuracy=%.4f balanced=%.4f f1_weighted=%.4f",
            report.accuracy,
            report.balanced_accuracy,
            report.f1_weighted,
        )
        return report._asdict()

    def _persist(
        self,
        model: XGBoostModel,
        extractor: TfidfFeatureExtractor,
        le: LabelEncoder,
        metrics: dict[str, Any],
        stats: dict[str, Any],
        val_rows: int,
        X_train: spmatrix,  # noqa: N803
        train_secs: float,
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
            "tfidf_max_label": cfg.tfidf_max_label,
            "tfidf_max_detail": cfg.tfidf_max_detail,
            "tfidf_max_char": cfg.tfidf_max_char,
            "random_state": cfg.random_state,
            "train_ratio": cfg.train_ratio,
            "train_rows": stats["train_rows"],
            "val_rows": val_rows,
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
        )
        logger.info("Stored version: %s", manifest.version)
        return manifest
