from src import config, data
from sklearn.metrics import mean_absolute_percentage_error
from lightgbm import LGBMRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import bayesian_search_forecaster, backtesting_forecaster
import logging
import joblib
import pandas as pd
import glob
import re
import numpy as np
from src.logger import logger


def get_path_model(iteration):
    return config.path_models / f"lgbm_forecaster_{iteration}.joblib"


def load_iteration(iteration):
    path_file = get_path_model(iteration)
    if not path_file.exists():
        logger.error(f"Iteration {iteration} does not exist!")
        return None
    return joblib.load(path_file)


def search_space(trial):
    search_space = {
        "num_leaves": trial.suggest_int("num_leaves", 10, 60),
        "max_depth": trial.suggest_int("max_depth", -1, 20),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 0.5),
        "lags": trial.suggest_categorical("lags", config.lags_consider),
    }
    return search_space


def get_last_model():
    list_files = glob.glob(str(config.path_models / "lgbm_forecaster_*.joblib"))
    if len(list_files) == 0:
        return -1, None
    iterations = [int(re.search(r"_(\d+)\.joblib", x).group(1)) for x in list_files]
    return max(iterations), joblib.load(config.path_models / f"lgbm_forecaster_{max(iterations)}.joblib")


class ForecasterLGBM:
    def __init__(self, iteration, end_train=None):
        logger.info(f"Initializing LGBM Forecaster {iteration}")
        self.forecaster = ForecasterAutoreg(
            regressor=LGBMRegressor(random_state=config.random_state, n_jobs=-1, verbose=-1), lags=1
        )
        self.iteration = iteration
        self.start_train = config.start_train
        self.end_train = end_train if end_train is not None else config.end_train
        self.is_tuned = False

    def make_idxs(self):
        X, y = data.load_all_data()

        idx_train = pd.date_range(self.start_train, self.end_train, freq="h")
        idx_train_noval = pd.date_range(self.start_train, self.end_train - config.delta_val, freq="h")
        idx_future = pd.date_range(self.end_train, X.index[-1], freq="h", inclusive="right")

        return idx_train, idx_train_noval, idx_future

    def save_to_file(self):
        path_to_save = get_path_model(self.iteration)
        logger.info(f"Saving Forecaster {self.iteration} to {path_to_save}.")
        joblib.dump(self, path_to_save)

    def tune(self):
        logger.info(f"Tuning Forecaster {self.iteration}")
        # Load the data
        X, y = data.load_all_data()
        idx_train, idx_train_noval, idx_future = self.make_idxs()
        results_hs_lgbm, frozen_trial_hs_lgbm = bayesian_search_forecaster(
            forecaster=self.forecaster,
            y=y.loc[idx_train],
            search_space=search_space,
            steps=config.predict_size,
            refit=config.refit_size,
            initial_train_size=len(idx_train_noval),
            metric="mean_absolute_percentage_error",
            exog=X.loc[idx_train],
            return_best=True,
            engine="optuna",
            random_state=config.random_state,
            verbose=False,
            n_trials=config.n_hyperparameters_trials,
        )
        logger.info("Tuning results: ")
        logger.info(f"Best parameters: {results_hs_lgbm.iloc[0]}")
        self.is_tuned = True
        logger.info(f"Model trained with data from {idx_train[0]} until {idx_train[-1]}!")
        self.save_to_file()

    def backtest(self):
        logger.info(f"Backtesting Forecaster {self.iteration}")
        # Load the data
        X, y, y_day_ahead = data.load_all_data()
        idx_train, idx_train_noval, idx_future = self.make_idxs()
        metric_lgbm, _ = backtesting_forecaster(
            self.forecaster,
            y.loc[idx_train],
            metric="mean_absolute_percentage_error",
            refit=False,
            initial_train_size=idx_train_noval.sum(),
            steps=config.predict_size,
            exog=X.loc[idx_train],
            random_state=config.random_state,
        )
        logger.info(f"Backtesting result: MAPE of {metric_lgbm}.")

    def predict(self):
        logger.info(f"Making predictions with Forecaster {self.iteration}")
        if not self.is_tuned:
            logger.info(f"Forecaster {self.iteration} not tuned!")
            self.tune()
        # Load the data
        X, y = data.load_all_data()
        idx_train, _, idx_future = self.make_idxs()
        n_steps = len(idx_future)
        if n_steps > config.refit_size * config.predict_size:
            logger.warn(f"Predicting {n_steps} hours (about {n_steps // 24} days), retraining might be necessary!")
        y_predicted = self.forecaster.predict(steps=n_steps, exog=X.loc[idx_future])
        mape = mean_absolute_percentage_error(y.loc[idx_future], y_predicted)
        logger.info(f"MAPE between data and predicted: {mape}")
        return mape, (y.loc[idx_future], y_predicted)

    def get_training(self):
        logger.info(f"Obtaining training estimation with Forecaster {self.iteration}")
        if not self.is_tuned:
            logger.info(f"Forecaster {self.iteration} not tuned!")
            self.tune()
        # Load the data
        X, y = data.load_all_data()
        idx_train, _, _ = self.make_idxs()
        X_train, y_train = self.forecaster.create_train_X_y(y=y.loc[idx_train], exog=X.loc[idx_train])
        y_train_pred = pd.Series(self.forecaster.regressor.predict(X_train), index=y_train.index)
        mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
        logger.info(f"Backtesting on training data with MAPE {np.round(mape_train,2)}!")
        return mape_train, (y_train, y_train_pred)

    def get_end_training(self):
        return self.end_train
