"""Abstract base for classification models."""

from abc import ABC, abstractmethod
from typing import Any, cast

import numpy as np
from scipy.sparse import spmatrix


class ClassifierBase(ABC):
    """Interface that every model implementation must satisfy."""

    def __init__(self, random_state: int = 42, **kwargs: Any):
        self.random_state = random_state
        self.model: Any = None
        self._ready = False

    @abstractmethod
    def fit(
        self,
        X_train: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
        y_train: np.ndarray[Any, np.dtype[Any]],
        X_val: spmatrix | np.ndarray[Any, np.dtype[Any]] | None = None,  # noqa: N803
        y_val: np.ndarray[Any, np.dtype[Any]] | None = None,
        sample_weight: np.ndarray[Any, np.dtype[Any]] | None = None,
    ) -> "ClassifierBase": ...

    @abstractmethod
    def predict(
        self,
        X: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    ) -> np.ndarray[Any, np.dtype[Any]]: ...

    def predict_proba(
        self,
        X: spmatrix | np.ndarray[Any, np.dtype[Any]],  # noqa: N803
    ) -> np.ndarray[Any, np.dtype[Any]]:
        if hasattr(self.model, "predict_proba"):
            return cast("np.ndarray[Any, np.dtype[Any]]", self.model.predict_proba(X))
        raise NotImplementedError("Probability predictions are not supported")

    def hyperparameters(self) -> dict[str, Any]:
        if hasattr(self.model, "get_params"):
            return cast("dict[str, Any]", self.model.get_params())
        return {}

    @property
    def ready(self) -> bool:
        return self._ready
