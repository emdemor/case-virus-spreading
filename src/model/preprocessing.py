import numpy as np
import pandas as pd
from src.base.logger import logging
from src.base.commons import load_pickle, load_yaml
from src.model.data import sanitize_features
from src.model.features import build_features

from src.global_variables import FEATURE_PARAMETERS_FILE


def one_hot_encode(
    X: pd.DataFrame,
    column: str,
    categories_to_drop: list or tuple = [],
    sep: str = "_",
) -> pd.DataFrame:
    """Encode categorical features as a one-hot numeric array.

    Parameters
    ----------
    X : pd.DataFrame
        Pandas dataframe with the column to be encoded
    column : str
        Categorical column to encode
    categories_to_drop : list or tuple
        List or tuple of column names to be neglected on encode process.
    sep : str, optional
        Character between prefix and category on the new column name, by default '_'

    Returns
    -------
    pd.DataFrame
        Dataframe with the column encoded
    """

    feature_parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    try:
        categories = feature_parameters[column]["encode"]["categories"]
        drop_columns = feature_parameters[column]["encode"]["drop_columns"] or []

        for category in categories:
            if category not in drop_columns:
                X[column + sep + category] = (X[column] == category).astype(int)

        X = X.drop(columns=column)

    except KeyError as err:
        error_mesage = (
            f"Column {err.__str__()} not found in file {FEATURE_PARAMETERS_FILE}."
        )
        logging.error(error_mesage)
        raise KeyError(error_mesage)

    return X


def discretize(X: pd.DataFrame, column: str, output_column_name: str) -> pd.DataFrame:
    """Discretize a continuos variable accorsing to predefined bins.

    Parameters
    ----------
    X : pd.DataFrame
        Pandas dataframe with the column to be encoded
    column : str
        Continuos column to be discretized
    output_column_name : str
        Name of discretized columns

    Returns
    -------
    pd.DataFrame
        Dataframe with the column discretized
    """

    feature_parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    try:
        generation_discretizer = feature_parameters[column]["discretize"]["bins"]

        bins = sorted([0] + [elem[1] for elem in generation_discretizer.values()])

        X[output_column_name] = pd.cut(
            X[column], bins=bins, right=False, labels=generation_discretizer.keys()
        )

        X = X.drop(columns=column)

        return X

    except KeyError as err:
        error_mesage = (
            f"Column {err.__str__()} not found in file {FEATURE_PARAMETERS_FILE}."
        )
        logging.error(error_mesage)
        raise KeyError(error_mesage)

    return X


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