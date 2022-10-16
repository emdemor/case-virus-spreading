import pandas as pd
from model.data import read_data_train_test
from base.commons import dump_json, dump_pickle, load_json, load_yaml
from base.logger import logging
from model.regressor import set_regressor
from optimizer.space import eval_parametric_space_dimension
from skopt import gp_minimize
from skopt.utils import use_named_args
from scipy.optimize import OptimizeResult
from optimizer.cross_validation import cross_validate_score
from sklearn.metrics import r2_score, mean_absolute_error
from global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
    MODEL_CONFIG_FILE,
)


def optimize_regressor():
    """ Trigger for model hyperparameter tunning """

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


def gaussian_process_optimization(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample: int = None,
) -> OptimizeResult:
    """Executes a Gaussian Process optimization for
    hyperparameter tunning

    Parameters
    ----------
    X_train : pd.DataFrame
        Features for training
    y_train : pd.Series
        response variable
    sample : int, optional
        Sample size to be used in each step for stocastic optimization, by default None

    Returns
    -------
    OptimizeResult
        Optimization results
    """

    logging.info("FUNCTION: gaussian_process_optimization")

    model_config = load_yaml(MODEL_CONFIG_FILE)

    model_parameters = load_json(model_config["parametric_space_path"])

    hyper_param = {
        hp["parameter"]: hp["estimate"] for hp in model_parameters["parametric_space"]
    }

    space = [
        eval_parametric_space_dimension(d) for d in model_parameters["parametric_space"]
    ]

    x0 = list(hyper_param.values())

    @use_named_args(space)
    def train_function(**params):

        estimator = set_regressor(model_config["model"], **params)

        cv_mae = cross_validate_score(
            X_train,
            y_train,
            estimator,
            scoring=mean_absolute_error,
            n_folds=model_config["opt_config"]["n_folds"],
            fit_params=model_config["fit_parameters"],
            sample=sample,
        )

        print(f"\n\n{80 * '#'}")
        print("params:", list(params.values()))
        print("mae:", cv_mae)
        print(f"{50 * '-'}")

        return cv_mae

    res_gp = gp_minimize(
        train_function,
        space,
        x0=x0,
        y0=None,
        random_state=model_config["opt_config"]["random_state"],
        verbose=model_config["opt_config"]["verbose"],
        n_calls=model_config["opt_config"]["n_calls"],
        n_random_starts=model_config["opt_config"]["n_random_starts"],
    )

    model_config["fit_parameters"]["eval_set"] = None

    return res_gp