from __future__ import annotations

import glob
import logging
import re
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from optuna.trial import Trial
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster, bayesian_search_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterRecursive
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor

from src import config as c
from src import data
from src.preprocessing import ExogBuilder, LinearlyInterpolateTS

logger = logging.getLogger(c.logger_name)


def get_path_model(name: str, iteration: int) -> Path:
    """Yields the path to the models for a given iteration and model name.

    Args:
        name (str): Model name.
        iteration (int): Iteration of the model.

    Returns:
        Path: Path where the model should be stored.
    """
    return c.path_models / f"{name}_forecaster_{iteration}.joblib"


def load_iteration(name: str, iteration: int) -> ForecasterRecursiveModel | None:
    """Loads a model at a given iteration.

    Args:
        name (str): Model name.
        iteration (int): Iteration of the model.

    Returns:
        ForecasterRecursiveModel | None: Loaded model.
    """
    path_file = get_path_model(name, iteration)
    if not path_file.exists():
        logger.error(f"Iteration {iteration} does not exist!")
        return None
    model: ForecasterRecursiveModel = joblib.load(path_file)
    return model


def search_space_lgbm(trial: Trial) -> dict:
    search_space = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "max_depth": trial.suggest_int("max_depth", 3, 16),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100),
        "lags": trial.suggest_categorical("lags", c.lags_consider),
    }
    return search_space


def search_space_xgb(trial: Trial) -> dict:
    search_space = {
        # "max_leaves": trial.suggest_int("max_leaves", 10, 100),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        "n_estimators": trial.suggest_int("n_estimators", 50, 600, step=50),
        "alpha": trial.suggest_float("alpha", 0.0, 0.5),
        "lambda": trial.suggest_float("lambda", 0.0, 0.5),
        "lags": trial.suggest_categorical("lags", c.lags_consider),
    }
    return search_space


window_features = RollingFeatures(
    stats=["mean", "mean", "mean", "min", "max"], window_sizes=[24, 24 * 7, 24 * 30, 24, 24]
)

SEARCH_SPACES = {"lgbm": search_space_lgbm, "xgb": search_space_xgb}


def get_last_model() -> tuple[int, ForecasterRecursiveModel | None]:
    """Get the latest model trained.

    Returns:
        tuple[int, ForecasterRecursiveModel | None]: Iteration and model last trained.
    """
    list_files = glob.glob(str(c.path_models / "*_forecaster_*.joblib"))
    if len(list_files) == 0:
        return -1, None
    searches = [re.search(r"_(\d+)\.joblib", x) for x in list_files]
    iterations = [int(search.group(1)) for search in searches if search is not None]
    if len(iterations) == 0:
        return -1, None
    max_iter = max(iterations)
    file_names = glob.glob(str(c.path_models / f"*_forecaster_{max_iter}.joblib"))
    if len(file_names) == 0:
        return -1, None
    file_name = file_names[0]
    return max_iter, joblib.load(file_name)


class ForecasterRecursiveModel:
    forecaster: ForecasterRecursive
    name: str

    def __init__(self, iteration: int, end_dev: str = None):
        self.iteration = iteration
        self.start_train = c.start_train
        self.end_dev = pd.to_datetime(end_dev if end_dev is not None else c.end_train_default, utc=True)
        self.end_train = pd.to_datetime(self.end_dev - c.delta_val, utc=True)
        self.start_dev = self.end_train + pd.Timedelta(hours=1)
        self.start_future = self.end_dev + pd.Timedelta(hours=1)
        self.is_tuned = False
        self.results_tuning = None
        self.metrics = ["mean_absolute_error", "mean_absolute_percentage_error"]
        self.best_params: dict | None = None
        self.best_lags: list[int] | None = None
        self.preprocessor = ExogBuilder(periods=c.periods, country_code=c.API_COUNTRY_CODE)

    def save_to_file(self) -> None:
        """Save model to file."""
        path_to_save = get_path_model(self.name, self.iteration)
        logger.info(f"Saving {self.name.upper()} Forecaster {self.iteration} to {path_to_save}.")
        joblib.dump(self, path_to_save)

    def _build_cv_dev(self, train_size: int) -> TimeSeriesFold:
        """Build cross validation time folds for hyperparameter tuning."""
        return TimeSeriesFold(
            steps=c.predict_size,
            # refit=c.refit_size,
            refit=False,
            initial_train_size=train_size,
            fixed_train_size=False,
            gap=0,
            skip_folds=None,
            allow_incomplete_fold=True,
        )

    def _build_cv_test(self, train_size: int) -> TimeSeriesFold:
        """Build cross validation time folds for backtesting."""
        return TimeSeriesFold(
            steps=c.predict_size,
            refit=c.refit_size,
            initial_train_size=train_size,
            fixed_train_size=False,
            gap=0,
            skip_folds=None,
            allow_incomplete_fold=True,
        )

    def fit_with_best(self) -> None:
        """After being tuned, fit the forecaster with the recorded best parameters and best results."""
        logger.info(f"Fitting {self.name.upper()} Forecaster {self.iteration} for predictions")

        # Check if model is tuned
        if self.best_params is None or self.best_lags is None:
            logger.warning("Model is not tuned! Starting tuning first...")
            self.tune()

        # Load the data
        y = data.load_timeseries()
        y = LinearlyInterpolateTS().apply(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        # Set parameters
        logger.info("Setting parameters...")
        self.forecaster.set_params(self.best_params)
        logger.info("Setting lags...")
        self.forecaster.set_lags(self.best_lags)

        # Fitting
        logger.info(
            f"Fitting over the whole training and dev sets {y.index[0].strftime(c.FORMAT_DATE_CSV)} - {self.end_dev.strftime(c.FORMAT_DATE_CSV)} ..."
        )
        self.forecaster.fit(y.loc[: self.end_dev], exog=X)
        logger.info("Training done!")

    def tune(self) -> None:
        """Tune the forecaster by performing a Bayesian search for optimal hyperparameters."""
        logger.info(f"Tuning {self.name.upper()} Forecaster {self.iteration}")

        # Load the data
        y = data.load_timeseries()
        # Preprocess the data
        y = LinearlyInterpolateTS().apply(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        # Perform bayesian search
        results, _ = bayesian_search_forecaster(
            forecaster=self.forecaster,
            y=y.loc[: self.end_dev],
            cv=self._build_cv_dev(len(y.loc[: self.end_train])),
            search_space=SEARCH_SPACES[self.name],
            metric=self.metrics,
            exog=X.loc[: self.end_dev],
            return_best=False,
            random_state=c.random_state,
            verbose=False,
            n_trials=c.n_hyperparameters_trials,
        )

        # Record results
        logger.info("Tuning results: ")
        results["name"] = self.name
        self.results_tuning = results
        logger.info(f"Best parameters:\n {results.iloc[0]}")
        self.best_params = results.iloc[0].params
        self.best_lags = results.iloc[0].lags

        # Fit the model with the best params and lags
        self.fit_with_best()
        logger.info(f"Model trained with data from {y.index.min()} until {self.end_dev}!")

        # Saved the model to file
        self.is_tuned = True
        self.save_to_file()

    def backtest(self) -> None:
        """Backtesting the forecaster on the test data."""
        logger.info(f"Backtesting {self.name.upper()} Forecaster {self.iteration}")

        # Load the data
        y = data.load_timeseries()
        y = LinearlyInterpolateTS().apply(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=y.index.max())

        # Fit forecaster over the entire train + dev sets
        self.fit_with_best()

        # Evaluate the model on the test data
        metrics, _ = backtesting_forecaster(
            self.forecaster,
            y,
            cv=self._build_cv_test(len(y.loc[: self.end_dev])),
            metric=self.metrics,
            exog=X,
            random_state=c.random_state,
        )
        logger.info(f"Backtesting results: {metrics.to_dict()}.")

    def predict(self, delta_predict: pd.Timedelta | None = None) -> tuple[dict, tuple[pd.Series, pd.Series]]:
        """Get the error and the prediction from the model."""
        logger.info(f"Making predictions with {self.name.upper()} Forecaster {self.iteration}")

        # Tune the model if it has not been done already
        if not self.is_tuned:
            self.tune()

        # Load the data
        y = data.load_timeseries()
        y = LinearlyInterpolateTS().apply(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=y.index.max())

        # Predict new data points
        if delta_predict is None or delta_predict > y.index.max() - self.start_future:
            end_predict = y.index.max()
        else:
            end_predict = self.start_future + delta_predict
        idx_future = pd.date_range(start=self.start_future, end=end_predict, freq="h")

        n_steps = len(idx_future)
        if n_steps > c.refit_size * c.predict_size:
            logger.info(f"Predicting {n_steps} hours (about {n_steps // 24} days), retraining might be necessary!")
        y_predicted = self.forecaster.predict(steps=n_steps, exog=X.loc[idx_future])

        # Evaluate solution
        metrics = dict()
        metrics["mape"] = mean_absolute_percentage_error(y.loc[idx_future], y_predicted)
        metrics["mae"] = mean_absolute_error(y.loc[idx_future], y_predicted)
        logger.info(
            f"Comparison between data and predicted until {end_predict}, MAPE: {metrics['mape']:.2f}, MAE: {metrics['mae']:.2f}"
        )
        return metrics, (y.loc[idx_future], y_predicted)

    def get_training(self) -> tuple[dict, tuple[pd.Series, pd.Series]]:
        """Get the error on the training dataset."""
        logger.info(f"Obtaining training estimation with Forecaster {self.iteration}")
        # Load the data
        y = data.load_timeseries()
        y = LinearlyInterpolateTS().apply(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        # Tune the model if it has not been done already
        if not self.is_tuned:
            self.tune()

        # Predict for training data
        X_train, y_train = self.forecaster.create_train_X_y(y=y.loc[: self.end_dev], exog=X.loc[: self.end_dev])
        y_train_pred = pd.Series(self.forecaster.regressor.predict(X_train), index=y_train.index)

        # Evaluate solution
        metrics = dict()
        metrics["mape"] = mean_absolute_percentage_error(y_train, y_train_pred)
        metrics["mae"] = mean_absolute_error(y_train, y_train_pred)
        logger.info(f"Backtesting on training data with MAPE {metrics['mape']:.2f}, MAE {metrics['mae']:.2f}!")
        return metrics, (y_train, y_train_pred)

    def get_error_forecast(self, delta_predict: pd.Timedelta | None = None) -> tuple[dict, tuple[pd.Series, pd.Series]]:
        """Get the error of the Entsoe forecast."""
        y = data.load_timeseries()
        y = LinearlyInterpolateTS().apply(y)

        y_forecast = data.load_timeseries_forecast()
        y_forecast = LinearlyInterpolateTS().apply(y_forecast)

        if delta_predict is None or delta_predict > y.index.max() - self.start_future:
            end_predict = y.index.max()
        else:
            end_predict = self.start_future + delta_predict
        idx_future = pd.date_range(start=self.start_future, end=end_predict, freq="h")

        metrics = dict()
        metrics["mape"] = mean_absolute_percentage_error(y.loc[idx_future], y_forecast.loc[idx_future])
        metrics["mae"] = mean_absolute_error(y.loc[idx_future], y_forecast.loc[idx_future])

        return metrics, (y.loc[idx_future], y_forecast.loc[idx_future])

    def package_prediction(self):
        """Method to package all the output for a 24h prediction"""
        metrics, (y_future, y_future_pred) = self.predict()
        metrics_train, (y_train, y_train_pred) = self.get_training()
        metrics_forecast, (_, y_forecast) = self.get_error_forecast()
        metrics_one_day, _ = self.predict(delta_predict=pd.Timedelta(hours=24))
        metrics_forecast_one_day, _ = self.get_error_forecast(delta_predict=pd.Timedelta(hours=24))
        return {
            "future_actual": y_future,
            "future_pred": y_future_pred,
            "train_actual": y_train,
            "train_pred": y_train_pred,
            "future_forecast": y_forecast,
            "metrics_future": metrics,
            "metrics_train": metrics_train,
            "metrics_forecast": metrics_forecast,
            "metrics_future_one_day": metrics_one_day,
            "metrics_forecast_one_day": metrics_forecast_one_day,
        }

    def get_feature_importance(self) -> pd.DataFrame | None:
        if self.name not in ["xgb", "lgbm"]:
            logger.error("Regressor does not support feature importance!")
            return None
        return self.forecaster.get_feature_importances()


class ForecasterRecursiveLGBM(ForecasterRecursiveModel):
    def __init__(self, iteration, end_dev=None):
        self.forecaster = ForecasterRecursive(
            regressor=LGBMRegressor(random_state=c.random_state, n_jobs=-1, verbose=-1),
            lags=12,
            window_features=window_features,
        )
        self.name = "lgbm"
        logger.info(f"Initializing {self.name.upper()} Forecaster {iteration}")
        super().__init__(iteration=iteration, end_dev=end_dev)


class ForecasterRecursiveXGB(ForecasterRecursiveModel):
    def __init__(self, iteration, end_dev=None):
        self.forecaster = ForecasterRecursive(
            regressor=XGBRegressor(random_state=c.random_state, n_jobs=-1), lags=12, window_features=window_features
        )
        self.name = "xgb"
        logger.info(f"Initializing {self.name.upper()} Forecaster {iteration}")
        super().__init__(iteration=iteration, end_dev=end_dev)
