from src.base.logger import logging
import unidecode
from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import unidecode
from src.base.commons import load_yaml, to_snake_case

PARAMETERS_CONFIG = load_yaml(filename="config/parameters.yaml")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return data


def generate_dummy_variables(
    dataframe: pd.DataFrame,
    column: str,
    categories: list or tuple or np.array,
    sep: str = "_",
    prefix: Any = "column",
):
    if prefix == "column":
        prefix = column

    for cat in categories:
        if prefix is not None:
            column_name = prefix + sep + cat
        else:
            column_name = cat

        dataframe = dataframe.assign(
            **{column_name: (dataframe[column] == cat).astype(float)}
        )

    return dataframe
