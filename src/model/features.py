from src.base.logger import logging
import unidecode
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import unidecode
from src.base.commons import load_yaml, to_snake_case

PARAMETERS_CONFIG = load_yaml(filename="config/parameters.yaml")


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Cosntruct new feature from raw dataset

    Parameters
    ----------
    X : pd.DataFrame
        Input dataset

    Returns
    -------
    pd.DataFrame
        New dataframe with constructed features
    """

    X["V1_estado_civil"] = np.where(
        (X["V1_estado_civil"] == "casado") | (X["V1_estado_civil"] == "divorciado"),
        "caso_ou_divorciado",
        X["V1_estado_civil"],
    )

    X["V1_tem_filhos"] = X["V1_qt_filhos"].clip(0, 1)

    X = X.drop(columns="V1_qt_filhos")

    return X
