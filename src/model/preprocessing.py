import numpy as np
import pandas as pd
from base.logger import logging
from base.commons import load_yaml
from model.features import build_features

from global_variables import FEATURE_PARAMETERS_FILE, PARAMETERS_FILE


def transform(X: pd.DataFrame) -> pd.DataFrame:
    """Apply preprocessing pipeline operations

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with the feature transformed.
    """

    parameters = load_yaml(PARAMETERS_FILE)

    input_columns = parameters["raw_dataframe_columns"]

    for feature in input_columns:
        if feature not in X.columns:
            error_mesage = f"Column '{feature}' not found in input dataframe."
            logging.error(error_mesage)
            raise KeyError(error_mesage)

    X = X[input_columns]

    X = apply_constant_imputes(X)

    X = build_features(X)

    X = discretize_features(X)

    X = encode_features(X)

    X = drop_columns(X)

    return X


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


def get_features_to_impute() -> dict:
    """Read feature parameters file and return
    a dict of features and values to constant impute

    Returns
    -------
    list
        Dict of features to cosntant impute
    """

    parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    return {
        x[0]: x[1]["constant_impute"]
        for x in list(
            filter(
                lambda x: "constant_impute" in x[1],
                [(key, value) for key, value in parameters.items()],
            )
        )
    }


def apply_constant_imputes(X: pd.DataFrame) -> pd.DataFrame:
    """Apply constant imputations defined on features parameter file

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with features imputed
    """

    imputations = get_features_to_impute()

    for column in imputations:
        X[column] = X[column].fillna(imputations[column])

    return X


def get_features_to_drop() -> list:
    """Read feature parameters file and returna list of feature to drop

    Returns
    -------
    list
        List of features to drop
    """

    parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    result = [
        x[0]
        for x in filter(
            lambda x: x[1]["drop"],
            filter(
                lambda x: "drop" in x[1],
                [(key, value) for key, value in parameters.items()],
            ),
        )
    ]

    return result


def drop_columns(X: pd.DataFrame) -> pd.DataFrame:
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

    columns_to_drop = get_features_to_drop()

    X = X.drop(columns=columns_to_drop)

    return X


def discretize(X: pd.DataFrame, column: str, output_column_name: str) -> pd.DataFrame:
    """Discretize a continuous variable according to predefined bins.

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

        if column != output_column_name:
            X = X.drop(columns=column)

        return X

    except KeyError as err:
        error_mesage = (
            f"Column {err.__str__()} not found in file {FEATURE_PARAMETERS_FILE}."
        )
        logging.error(error_mesage)
        raise KeyError(error_mesage)

    return X


def discretize_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply discretization operation according to the features
    in parameter_features yaml file.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with features discretized
    """

    for cont, disc in get_discretized_output_names().items():
        if cont in X.columns:
            X = discretize(X, cont, disc)
        else:
            error_mesage = f"Column '{cont}' not found in input dataframe."
            logging.error(error_mesage)
            raise KeyError(error_mesage)

    return X


def get_discretized_output_names() -> dict:
    """Read feature parameters file and return a dict of
    features to be discretized related to the output column name.

    Returns
    -------
    dict
        Dict of features to discretize and the output column name
    """

    parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    result = {
        x[0]: x[1]["discretize"]["output_column"]
        for x in filter(
            lambda x: "discretize" in x[1],
            [(key, value) for key, value in parameters.items()],
        )
    }

    return result


def encode_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply one-hot-encode operation according to the features
    in parameter_features yaml file.

    Parameters
    ----------
    X : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Dataframe with features encoded
    """

    features_to_encode = get_features_to_encode()

    for feature in features_to_encode:
        if feature in X.columns:
            X = one_hot_encode(X, feature)
        else:
            error_mesage = f"Column '{feature}' not found in input dataframe."
            logging.error(error_mesage)
            raise KeyError(error_mesage)

    return X


def get_features_to_encode() -> list:
    """Read feature parameters file and return a list of features to encode

    Returns
    -------
    list
        List of features to encode
    """

    parameters = load_yaml(FEATURE_PARAMETERS_FILE)

    result = [
        x[0]
        for x in filter(
            lambda x: "encode" in x[1],
            [(key, value) for key, value in parameters.items()],
        )
    ]

    return result
