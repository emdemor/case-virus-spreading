import pandas as pd
import numpy as np
import yaml

from src.base.logger import logging
from src.base.commons import dump_json, dump_pickle, load_json, load_yaml
from src.model.regressor import set_regressor
from src.model.data import read_data_train_test
from src.optimizer import gaussian_process_optimization


from sklearn.metrics import mean_absolute_error, r2_score

from src.global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
    MODEL_CONFIG_FILE,
)


def optimize_regressor():
    """ Trigger for model hyperparamter tunning """

    filepaths = load_yaml(filename=FILEPATHS_FILE)
    model_config = load_yaml(MODEL_CONFIG_FILE)
    model_parameters = load_json(model_config["parametric_space_path"])

    hyper_param = {
        hp["parameter"]: hp["estimate"] for hp in model_parameters["parametric_space"]
    }

    X_train, X_test, y_train, y_test = read_data_train_test()

    optimizer = gaussian_process_optimization(
        X_train, y_train, sample=model_config["cv_optmize_sample"]
    )

    proposal_params = dict(zip(hyper_param.keys(), optimizer.x))

    logging.info(f"Proposal parameters: {proposal_params}")

    logging.info("Set regressor with proposed parameters")
    proposal_regessor = set_regressor(model_config["model"], **proposal_params)
    proposal_regessor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)
    y_test_pred = proposal_regessor.predict(X_test)
    mae_proposal = mean_absolute_error(y_test, y_test_pred)
    r2_proposal = r2_score(y_test, y_test_pred)

    logging.info("mae_proposed = {:.6f}".format(mae_proposal))
    logging.info("R2_proposed = {:.6f}".format(r2_proposal))

    mae_current = model_parameters["metric"]["value"]

    logging.info("mae_current = {:.6f}".format(mae_current))
    logging.info("mae_proposal = {:.6f}".format(mae_proposal))

    if mae_proposal < mae_current:

        logging.info("Optimization found a better model.")
        logging.info("Update model with new parameters")
        logging.info("Update {} file".format(model_config["parametric_space_path"]))

        for i, param in enumerate(model_parameters["parametric_space"]):
            param.update({"estimate": optimizer.x[i]})

        model_parameters["metric"].update({"value": mae_proposal})

        dump_json(model_parameters, model_config["parametric_space_path"], indent=4)

        dump_pickle(
            proposal_regessor,
            filepaths["model_regressor_path"].format(model=model_config["model"]),
        )

    else:
        logging.info(
            "Optimization was not capable to find a better model. Keeping with the old model."
        )


if __name__ == "__main__":
    try:
        optimize_regressor()
    except Exception as err:
        logging.error(err)
        raise err