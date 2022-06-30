import os
import click
import pandas as pd
import numpy as np
from src.base.commons import load_yaml

from src.base.logger import logging
from src.model.regressor import get_regressor
from src.model import preprocessing, data
from src.global_variables import FILEPATHS_FILE

FILEPATHS = load_yaml(filename=FILEPATHS_FILE)

OUTPUT_FILE = os.path.join(
    FILEPATHS["predicted_directory_path"], "prob_predictions.csv"
)


@click.command()
@click.option(
    "--output_file",
    "-o",
    default=OUTPUT_FILE,
    help="Path to the output file",
    type=str,
)
def predict(output_file):

    output_file = output_file

    logging.info("Import feature to predict")
    X_pred = pd.read_parquet(
        os.path.join(FILEPATHS["interim_directory_path"], "data_modelling.parquet")
    ).drop(columns="prob_V1_V2")

    logging.info("Applying preprocessing pipeline")
    X_pred_transf = preprocessing.transform(X_pred)

    logging.info("Importing model")
    model = get_regressor()

    logging.info("Perdicting probabilities")
    y_pred = model.predict(X_pred_transf)
    df_predictions = pd.DataFrame({"prob_V1_V2_predicted": y_pred}, index=X_pred.index)

    logging.info(f"Export predictions to {output_file}")
    df_predictions.to_csv(output_file)


if __name__ == "__main__":
    try:
        predict()
    except Exception as err:
        logging.error(err)
        raise err