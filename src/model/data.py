import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.base.logger import logging
from src.base.commons import load_yaml, read_csv
from src.global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
)


def prepare_datasets() -> None:
    """ Pipeline for data preparation """

    # Import parameters file
    parameters = load_yaml(PARAMETERS_FILE)

    logging.info("Get full dataset")
    df_raw = make_datasets()

    logging.info("Split into modelling and predict datasets")
    data_modelling, data_predict = modelling_predict_split(df_raw)

    logging.info("Split modelling data into train and test")
    data_train, data_validation = train_test_split(
        data_modelling, **parameters["train_test_split"]
    )

    logging.info("Persists interim dataframes")
    persist_interim_table(data_modelling, "data_modelling")
    persist_interim_table(data_predict, "data_predict")
    persist_interim_table(data_train, "data_train")
    persist_interim_table(data_validation, "data_validation")


def make_datasets() -> pd.DataFrame:
    """Read data from connection and individuals, merge infective and infected
    individual data into connection table and return the resulting dataframe.

    Returns
    -------
    pd.DataFrame
        Connection dataframe with infective and infected personal information.
    """

    raw_datasets = load_datasets()

    df_raw_conexoes = raw_datasets["connections"]

    df_raw_individuos = raw_datasets["individuals"]

    df_raw = (
        df_raw_conexoes.merge(
            df_raw_individuos.rename(
                columns={col: "V1_" + col for col in df_raw_individuos.columns}
            ),
            left_on="V1",
            right_on="V1_name",
            how="left",
        )
        .merge(
            df_raw_individuos.rename(
                columns={col: "V2_" + col for col in df_raw_individuos.columns}
            ),
            left_on="V2",
            right_on="V2_name",
            how="left",
        )
        .drop(columns=["V1_name", "V2_name"])
        .set_index(["V1", "V2"])
    )

    return df_raw


def load_datasets() -> dict:
    """Import the individual and connection tables.

    Returns
    -------
    dict
        A dictionary with keys `connections` and `individuals` and values
        given by dataframes of connections and individuals respectivelly.
    """

    filepaths = load_yaml("config/filepaths.yaml")

    df_raw_conexoes = read_csv(filepaths["data_raw_conexoes_path"], sep=";")

    df_raw_individuos = read_csv(filepaths["data_raw_individuos_path"], sep=";")

    return {
        "connections": df_raw_conexoes,
        "individuals": df_raw_individuos,
    }


def modelling_predict_split(
    raw_dataset: pd.DataFrame, target_variable: str = "prob_V1_V2"
) -> tuple:
    """Splits the data according to the response variable.
    When the target is null, it is considered as a dataset
    for prediction and otherwise as a dataset for modeling.

    Parameters
    ----------
    raw_dataset : pd.DataFrame
        Dataframe with infective and infected personal information and connection.
    target_variable : str, optional
        Column in dataframe with the target variable, by default 'prob_V1_V2'

    Returns
    -------
    tuple
        A tuple of data_modelling and data_predict respectivelly.
    """

    data_modelling = raw_dataset.loc[raw_dataset[target_variable].notna()].copy()

    data_predict = raw_dataset.loc[raw_dataset[target_variable].isna()].copy()

    return data_modelling, data_predict


def persist_interim_table(table: pd.DataFrame, filename: str, *args, **kwargs) -> None:
    """Persist a dataframe to a parquet file in the interim data folder.

    Parameters
    ----------
    table : pd.DataFrame
        Pandas dataframe to be persisted.
    filename : str
        Path to the output parquet file
    """

    filepaths = load_yaml(FILEPATHS_FILE)

    interim_path = os.path.join(
        filepaths["interim_directory_path"],
        filename + ".parquet",
    )

    try:
        table.to_parquet(interim_path, *args, **kwargs)
    except Exception as err:
        logging.error(err)
        raise err


def persist_processed_table(
    table: pd.DataFrame, filename: str, *args, **kwargs
) -> None:
    """Persist a dataframe to a parquet file in the processed data folder.

    Parameters
    ----------
    table : pd.DataFrame
        Pandas dataframe to be persisted.
    filename : str
        Path to the output parquet file
    """

    filepaths = load_yaml(FILEPATHS_FILE)

    processed_path = os.path.join(
        filepaths["processed_directory_path"],
        filename + ".parquet",
    )

    try:
        table.to_parquet(processed_path, *args, **kwargs)
    except Exception as err:
        logging.error(err)
        raise err


def read_data_train_test():

    filepaths = load_yaml(FILEPATHS_FILE)

    X_train = pd.read_parquet(
        os.path.join(
            filepaths["processed_directory_path"],
            "X_train_transf.parquet",
        )
    )

    y_train = pd.read_parquet(
        os.path.join(
            filepaths["processed_directory_path"],
            "y_train.parquet",
        )
    ).iloc[:, 0]

    # Dados de Validação
    X_validation = pd.read_parquet(
        os.path.join(
            filepaths["processed_directory_path"],
            "X_validation_transf.parquet",
        )
    )

    y_validation = pd.read_parquet(
        os.path.join(
            filepaths["processed_directory_path"],
            "y_validation.parquet",
        )
    ).iloc[:, 0]

    return X_train, X_validation, y_train, y_validation
