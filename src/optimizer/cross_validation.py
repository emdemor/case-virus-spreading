from typing import Callable
import xgboost
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold, StratifiedKFold


def cross_validate_score(
    X: pd.DataFrame,
    y: pd.Series,
    estimator: RegressorMixin,
    scoring: Callable,
    verbose: int = 1,
    n_folds: int = 5,
    random_state: int = 42,
    fit_params: dict = {},
    minimize: bool = True,
    sample: int = None,
) -> float:
    """Evaluates cross-validation score

    Parameters
    ----------
    X : pd.DataFrame
        Model features
    y : pd.Series
        Response variable
    estimator : RegressorMixin
        Regressor model instance
    scoring : Callable
        Funciton defining the model metric
    verbose : int, optional
        Level of logging, by default 1
    n_folds : int, optional
        The number of folds in cross-validation, by default 5
    random_state : int, optional
        Random seed, by default 42
    fit_params : dict, optional
        Parameters to be passed on fit method, by default {}
    minimize : bool, optional
        If the metric is better when is lower, by default True
    sample : int, optional
        Sample size to be used in each step for stocastic optimization, by default None

    Returns
    -------
    float
        Value of metric mean
    """

    scores = []

    if sample is not None:
        data = X.assign(y=y.values).sample(sample)
    else:
        data = X.assign(y=y.values)

    data = data.reset_index(drop=True)

    data["fold"] = generate_folds(data, n_folds=n_folds, random_state=random_state)

    iterator = (
        range(n_folds) if verbose < 1 else tqdm(range(n_folds), desc="Cross validation")
    )

    for fold in iterator:

        # Separando os dados de treinamento para essa fold
        train_data = data[data["fold"] != fold].copy()

        # Separando os dados de teste para esse fold
        test_data = data[data["fold"] == fold].copy()

        X_1 = train_data.drop(columns=["fold", "y"]).values

        X_2 = test_data.drop(columns=["fold", "y"]).values

        y_1 = train_data["y"].values

        y_2 = test_data["y"].values

        if estimator.__class__ in [
            xgboost.sklearn.XGBRegressor,
            xgboost.sklearn.XGBClassifier,
        ]:
            fit_params["eval_set"] = [(X_2, y_2)]

        try:
            estimator.fit(X_1, y_1, **fit_params)
        except:
            estimator.fit(X_1, y_1, **fit_params)

        scores.append(scoring(y_2, estimator.predict(X_2)))

    if minimize:
        avg_score = np.mean(scores)
    else:
        avg_score = -np.mean(scores)

    return avg_score


def generate_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy().reset_index(drop=True)

    # Instaciando o estritificador
    kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(kf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[temp[temp.index.isin(validation_index)].index, "fold"] = fold_index

    return temp["fold"].astype(int)


def generate_stratified_folds(train, n_folds=5, shuffle=True, random_state=42):

    temp = train.copy().reset_index(drop=True)

    # Instaciando o estritificador
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)

    # Gerando os index com os folds
    stratified_folds = list(skf.split(X=temp.drop(columns="y"), y=temp["y"]))

    for fold_index in range(n_folds):

        train_index, validation_index = stratified_folds[fold_index]

        temp.loc[temp[temp.index.isin(validation_index)].index, "fold"] = fold_index

    return temp["fold"].astype(int)