from sklearn.base import RegressorMixin
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from src.base.commons import load_json, load_pickle, load_yaml
from src.global_variables import (
    FEATURE_PARAMETERS_FILE,
    FILEPATHS_FILE,
    PARAMETERS_FILE,
    MODEL_CONFIG_FILE,
)


def set_regressor(model: RegressorMixin, **regressor_args) -> RegressorMixin:
    """Instantiate regressor class according to the
    parameters specified in MODEL_CONFIG_FILE

    Parameters
    ----------
    model : RegressorMixin
        Description of regressor model

    Returns
    -------
    RegressorMixin
        A regressor instance
    """

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
    """Imports the trained model according to MODEL_CONFIG_FILE

    Returns
    -------
    RegressorMixin
        A regressor instance
    """

    filepaths = load_yaml(filename=FILEPATHS_FILE)
    model_config = load_yaml(MODEL_CONFIG_FILE)

    regressor = load_pickle(
        filepaths["model_regressor_path"].format(model=model_config["model"])
    )
    return regressor