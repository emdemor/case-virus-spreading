from __future__ import annotations

import numpy as np
import pandas as pd
from src.base.logger import loggin
from src.base.commons import load_pickle, load_yaml
from src.model.data import sanitize_features
from src.model.features import build_features


def apply_preprocess(preprocessor, X: pd.DataFrame):

    X = preprocessor.transform(X)

    # Custom steps...

    return X


def preprocess_transform(X: pd.DataFrame) -> pd.DataFrame:

    filepaths = load_yaml(filename="config/filepaths.yaml")

    preprocessor = load_pickle(filepaths["model_preprocessor_path"])

    X = sanitize_features(X)
    X = build_features(X)
    X = apply_preprocess(preprocessor, X)

    return X