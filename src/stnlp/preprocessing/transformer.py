"""Scikit-learn transformer wrapping the same preprocessing used by inference."""

from __future__ import annotations

from typing import Iterable

from sklearn.base import BaseEstimator, TransformerMixin

from stnlp.preprocessing.text import preprocess_text


class VietnameseTextTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stops: bool = True):
        self.remove_stops = remove_stops

    def fit(self, X: Iterable[str], y=None):
        return self

    def transform(self, X: Iterable[str]) -> list[str]:
        return [
            preprocess_text(value, remove_stops=self.remove_stops)
            for value in X
        ]
