"""End-to-end model training orchestration."""

import logging
import time

import pandas as pd
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
        """Run the full pipeline and return the saved bundle's manifest."""
        cfg = self.settings
        set_seed(cfg.random_state)

        # 1 — Ingest data
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

        # 2 — Temporal split
        logger.info("Splitting (temporal, ratio=%.2f) …", cfg.train_ratio)
        train_df, val_df = split_by_date(df, train_ratio=cfg.train_ratio)
        stats = temporal_partition_stats(train_df, val_df)
        logger.info(
            "Train: %d | Val: %d | Classes: %d",
            stats["train_rows"],
            stats["val_rows"],
            stats["train_n_classes"],
        )

        # 3 — Encode labels
        le = LabelEncoder()
        y_train = le.fit_transform(train_df["target"])
        baseline_accuracy = float(pd.Series(y_train).value_counts(normalize=True).iloc[0])
        known_mask = val_df["target"].isin(le.classes_)
        val_known = val_df[known_mask]
        y_val = le.transform(val_known["target"])
        if len(val_known) < len(val_df):
            logger.warning(
                "Excluded %d val rows with unseen classes",
                len(val_df) - len(val_known),
            )

        # 4 — Build features
        logger.info("Building features …")
        t0 = time.time()
        extractor = TfidfFeatureExtractor(
            label_vocab_size=cfg.tfidf_max_label,
            detail_vocab_size=cfg.tfidf_max_detail,
            char_vocab_size=cfg.tfidf_max_char,
        )
        domain_engine = DomainFeatureEngine(cfg.feature_profile)
        X_train = assemble_feature_matrix(train_df, extractor, domain_engine, fit=True)
        X_val = assemble_feature_matrix(val_known, extractor, domain_engine, fit=False)
        logger.info("Feature shape: %s (%.1fs)", X_train.shape, time.time() - t0)

        # 5 — Train model
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

        # 6 — Evaluate
        logger.info("Evaluating on validation set …")
        y_hat = model.predict(X_val)
        report = evaluate_predictions(y_val, y_hat)
        metrics_dict = report._asdict()
        logger.info(
            "Val accuracy=%.4f balanced=%.4f f1_weighted=%.4f",
            report.accuracy,
            report.balanced_accuracy,
            report.f1_weighted,
        )

        # 7 — Persist artefacts
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
            "val_rows": len(val_known),
            "num_categories": len(le.classes_),
            "train_seconds": train_secs,
            "env": get_reproducibility_info(),
        }
        manifest = store.save(
            model=model,
            text_extractor=extractor,
            label_encoder=le,
            metrics=metrics_dict,
            config=run_config,
            n_features=X_train.shape[1],
        )
        logger.info("Stored version: %s", manifest.version)
        return manifest, baseline_accuracy, len(le.classes_)
