import os
import click

import pandas as pd

from src.model import preprocessing, data, regressor

from src.base.commons import dump_pickle, load_yaml
from src.base.logger import logging
from src.global_variables import FILEPATHS_FILE, MODEL_CONFIG_FILE

from cli.optimize import optimize_regressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


@click.command()
@click.option(
    "--config_file",
    "--config",
    "-c",
    default=MODEL_CONFIG_FILE,
    help="YAML file with model config",
    type=str,
)
@click.option(
    "--optimize",
    "--opt",
    "-o",
    default=False,
    help="Hyperparameter tunnig optimization",
    type=bool,
)
def train(config_file, optimize):

    # Importing addresses
    filepaths = load_yaml(FILEPATHS_FILE)
    model_config = load_yaml(config_file)

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

    if optimize:
        logging.info("Hyperparameter tunning")
        optimize_regressor()

    X_train, X_val, y_train, y_val = data.read_data_train_test()

    logging.info("Training model with the optimized parameters")
    model = regressor.get_regressor()
    model.fit(X_train, y_train)

    logging.info("Validation Metrics")
    y_val_pred = model.predict(X_val)
    logging.info("r2_score = {:.6f}".format(r2_score(y_val, y_val_pred)))
    logging.info("mae = {:.6f}".format(mean_absolute_error(y_val, y_val_pred)))
    logging.info("rmse = {:.6f}".format(mean_squared_error(y_val, y_val_pred) ** 0.5))

    logging.info("Dumping model pickle")
    dump_pickle(
        model,
        filepaths["model_regressor_path"].format(model=model_config["model"]),
    )


if __name__ == "__main__":

    try:
        train()
    except Exception as err:
        logging.error(err)
        raise err
