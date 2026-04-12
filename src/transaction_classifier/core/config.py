"""Application configuration loaded from environment variables."""

from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings store, populated via TXCLS_* environment variables."""

    # --- Logging -----------------------------------------------------------
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # --- Data ingestion ----------------------------------------------------
    data_path: str = "data/sample.csv"
    client_registry_path: str = "clients.yaml"
    pg_row_limit: int = 500_000

    # --- Label derivation & filtering --------------------------------------
    target_length: int = 6
    min_class_samples: int = 10

    # --- TF-IDF vocabulary sizes -------------------------------------------
    tfidf_max_label: int = 4000
    tfidf_max_detail: int = 4000
    tfidf_max_char: int = 1000

    # --- XGBoost defaults --------------------------------------------------
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    patience: int | None = 40
    max_bin: int = 256
    device: str = "cpu"
    random_state: int = 42
    train_ratio: float = 0.80

    # --- Quality gate ------------------------------------------------------
    min_lift: float = 0.20

    # --- Feature profile ----------------------------------------------------
    feature_profile: str = "config/profiles/french_treasury.yaml"

    # --- Artifact storage --------------------------------------------------
    artifact_dir: str = "models"

    # --- Serving -----------------------------------------------------------
    sandbox_mode: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    default_top_k: int = 3
    batch_limit: int = 100
    reload_poll_secs: int = 30

    # --- Auth --------------------------------------------------------------
    api_keys: list[str] = []
    admin_api_keys: list[str] = []

    model_config = {"env_prefix": "TXCLS_"}
