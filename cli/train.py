import os
import pandas as pd

from src.model import preprocessing, data
from src.base.commons import load_yaml
from src.base.logger import logging
from src.global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
)


def train():

    # Importing addresses
    filepaths = load_yaml(FILEPATHS_FILE)

    # Preparing datasets
    data.prepare_datasets()

    logging.info("Importing train and validation interim tables")

    data_train = pd.read_parquet(
        os.path.join(
            filepaths["interim_directory_path"],
            "data_train.parquet",
        )
    )

    data_validation = pd.read_parquet(
        os.path.join(
            filepaths["interim_directory_path"],
            "data_validation.parquet",
        )
    )

    logging.info("Splitting target from features")
    X_train = data_train.drop(columns="prob_V1_V2")
    y_train = data_train["prob_V1_V2"]
    X_validation = data_validation.drop(columns="prob_V1_V2")
    y_validation = data_validation["prob_V1_V2"]

    logging.info("Applying preprocessing transformations")
    X_train_transf = preprocessing.transform(X_train)
    X_validation_transf = preprocessing.transform(X_validation)

    logging.info("Persisting transformed features")
    data.persist_processed_table(X_train_transf, "X_train_transf")
    data.persist_processed_table(X_validation_transf, "X_validation_transf")

    logging.info("Persisting response variable")
    data.persist_processed_table(y_train.to_frame(), "y_train")
    data.persist_processed_table(y_validation.to_frame(), "y_validation")

    pass


if __name__ == "__main__":
    train()
