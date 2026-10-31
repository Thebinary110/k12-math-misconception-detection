"""Shared data loading, MAP@3 and CV split for the K-12 misconception competition."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
N_FOLDS = 5
SEED = 42


def load():
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "test.csv")
    taxonomy = pd.read_csv(DATA / "taxonomy_v1.csv")
    return train, test, taxonomy


def classes(train, taxonomy):
    """All taxonomy codes plus NONE, so codes unseen in a fold still have a column."""
    return sorted(set(taxonomy["id"]) | set(train["root_code"]) | {"NONE"})


def folds(train, n_folds=N_FOLDS, seed=SEED):
    """Row-level stratified folds: test shares questions with train, so the split keeps that overlap.
    Codes with fewer examples than folds are grouped into one stratum."""
    y = train["root_code"].where(train["root_code"].map(train["root_code"].value_counts()) >= n_folds, "RARE")
    return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(train, y))


def top3(P, cls):
    idx = np.argsort(-P, axis=1)[:, :3]
    return [[cls[j] for j in row] for row in idx]


def map3(y_true, preds):
    score = 0.0
    for t, p in zip(y_true, preds):
        for k, c in enumerate(p[:3]):
            if c == t:
                score += 1.0 / (k + 1)
                break
    return score / len(y_true)
