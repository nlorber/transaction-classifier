"""TF-IDF feature extraction for transaction text fields."""

import pandas as pd
from scipy.sparse import hstack, spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

from ..data.preprocessor import normalize_text, strip_html


class TfidfFeatureExtractor:
    """Builds sparse TF-IDF matrices from the *description*, *remarks*, and combined text.

    Three independent vectorisers:
    - **label** (description): word-level n-grams
    - **detail** (remarks): word-level n-grams
    - **character**: character-level n-grams on concatenated text
    """

    def __init__(
        self,
        # Defaults here mirror Settings.tfidf_max_* in core/config.py.
        # The training pipeline passes config values explicitly; the serving
        # loader overwrites vectorizers from disk, so these only matter as
        # a fallback.
        label_vocab_size: int = 4000,
        detail_vocab_size: int = 4000,
        char_vocab_size: int = 1000,
        label_ngrams: tuple[int, int] = (1, 2),
        detail_ngrams: tuple[int, int] = (1, 2),
        char_ngrams: tuple[int, int] = (3, 5),
        min_df: int = 3,
        max_df: float = 0.95,
    ):
        self.vec_label = TfidfVectorizer(
            max_features=label_vocab_size,
            ngram_range=label_ngrams,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.vec_detail = TfidfVectorizer(
            max_features=detail_vocab_size,
            ngram_range=detail_ngrams,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.vec_char = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngrams,
            max_features=char_vocab_size,
            min_df=min_df,
            sublinear_tf=True,
        )
        self._fitted = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _clean_columns(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        label_text = df["description"].fillna("").apply(normalize_text)
        detail_text = df["remarks"].fillna("").apply(strip_html).apply(normalize_text)
        merged = label_text + " " + detail_text
        return label_text, detail_text, merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TfidfFeatureExtractor":
        """Learn vocabularies from training data."""
        lbl, dtl, merged = self._clean_columns(df)
        self.vec_label.fit(lbl)
        self.vec_detail.fit(dtl)
        self.vec_char.fit(merged)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> spmatrix:
        """Produce a sparse feature matrix from previously learned vocabularies."""
        if not self._fitted:
            raise ValueError("Call fit() before transform()")

        lbl, dtl, merged = self._clean_columns(df)
        return hstack(
            [
                self.vec_label.transform(lbl),
                self.vec_detail.transform(dtl),
                self.vec_char.transform(merged),
            ]
        )

    def fit_transform(self, df: pd.DataFrame) -> spmatrix:
        """Convenience: fit and transform in a single pass."""
        self.fit(df)
        return self.transform(df)

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature names across all three vectorisers."""
        if not self._fitted:
            raise ValueError("Extractor has not been fitted yet")
        names: list[str] = []
        names.extend(f"desc_{n}" for n in self.vec_label.get_feature_names_out())
        names.extend(f"rem_{n}" for n in self.vec_detail.get_feature_names_out())
        names.extend(f"chr_{n}" for n in self.vec_char.get_feature_names_out())
        return names

    @classmethod
    def from_vectors(
        cls,
        vectors: dict[str, TfidfVectorizer],
    ) -> "TfidfFeatureExtractor":
        """Reconstruct a fitted extractor from previously saved vectorizers."""
        instance = cls.__new__(cls)
        instance.vec_label = vectors["label"]
        instance.vec_detail = vectors["detail"]
        instance.vec_char = vectors["char"]
        instance._fitted = True
        return instance

    @property
    def vectorizer_dict(self) -> dict[str, TfidfVectorizer]:
        """Access the underlying fitted vectoriser objects."""
        return {
            "label": self.vec_label,
            "detail": self.vec_detail,
            "char": self.vec_char,
        }
