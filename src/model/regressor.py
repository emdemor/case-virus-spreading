from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.base.commons import load_pickle, load_yaml
from src.global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
    MODEL_CONFIG_FILE,
)


def set_regressor(model, **regressor_args):

    model_config = load_yaml(MODEL_CONFIG_FILE)

    static_parameters = model_config["static_parameters"]

    model = model_config["model"]

    if model == "xgboost":
        model = XGBRegressor(**static_parameters, **regressor_args)
        return model

    if model == "lightgbm":
        model = LGBMRegressor(**static_parameters, **regressor_args)
        return model


def get_regressor():
    filepaths = load_yaml(filename=FILEPATHS_FILE)
    model_config = load_yaml(MODEL_CONFIG_FILE)

    regressor = load_pickle(
        filepaths["model_regressor_path"].format(model=model_config["model"])
    )
    return regressor


def get_model_parameters():
    model_config = load_yaml(MODEL_CONFIG_FILE)

    model_parameters = {}
    model_parameters["model"] = model_config["model"]
    model_parameters["metric"] = model_config["metric"]
    model_parameters["parametric_space"] = model_config["parametric_space"]

    return model_parameters