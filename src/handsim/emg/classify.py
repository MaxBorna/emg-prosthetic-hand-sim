"""Baseline classifiers. Standardisation matters: WL is ~100x larger than MAV."""

from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def make_classifier(name: str = "lda") -> Pipeline:
    if name == "lda":
        clf = LinearDiscriminantAnalysis()
    elif name == "svm":
        clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced")
    else:
        raise ValueError(f"unknown classifier {name!r}")
    return make_pipeline(StandardScaler(), clf)
