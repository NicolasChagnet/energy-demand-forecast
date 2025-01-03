import argparse
import logging
import time
from logging.handlers import RotatingFileHandler

import pandas as pd
from dotenv import dotenv_values

from src import config, data, models
from src.figure import PredictionFigure
from src.models import ForecasterRecursiveModel

list_models = {"lgbm": models.ForecasterRecursiveLGBM, "xgb": models.ForecasterRecursiveXGB}


def merge_build_manual() -> None:
    """Merge raw files together."""
    data.merge_raw_files()


def download_new_data(api_key: str, start: str | None = None, end: str | None = None, force: bool = False) -> None:
    """Download new data from the Entsoe API for a given period."""
    data.merge_raw_files()
    logger.info("Downloading new data...")
    if start is None:
        current_data = data.load_full_data()
        start_date = current_data.index[-1] + pd.Timedelta(hours=1)
    else:
        start_date = pd.to_datetime(start, format=config.FORMAT_DATE_API, utc=True)
    if end is None:
        end_date = pd.Timestamp.utcnow().floor("d")  # Defaults to today, rounded down to midnight
    else:
        end_date = pd.to_datetime(end, format=config.FORMAT_DATE_API, utc=True)
    if ((end_date - start_date).total_seconds() // 3600) < 24 and not force:
        logger.info("Last download less than 24h ago, aborting download...")
        return None

    # Retry loop in case there is an error
    retry_counter = 0
    while retry_counter < 5:
        status = data.download_data(api_key, start_date, end_date)
        if status:
            break
        retry_counter += 1
        time.sleep(5)
    data.merge_raw_files()


def get_model_prediction() -> dict | None:
    """Get the prediction from the latest model trained."""
    n_iteration, model = models.get_last_model()
    if n_iteration < 0 or model is None:
        logger.error("No model found, train a model first!")
        return None
    logger.info("Making predictions using previous model...")
    return model.package_prediction()


def train_new_model(model_class: type[ForecasterRecursiveModel], n_iteration: int) -> None:
    """Train a new model of a given model class."""
    logger.info("Training new model...")
    current_data = data.load_full_data()
    latest_idx = current_data.index[-1]
    end_train_cutoff = latest_idx - pd.Timedelta(days=1)
    model = model_class(n_iteration, end_dev=end_train_cutoff, train_size=config.train_size)
    model.tune()


def handle_training(model_type: str | None = None, force: bool = False) -> None:
    """If the current model is old enough, request a new model to be trained."""
    # Choosing the model
    model_type = model_type or "lgbm"
    model_class = list_models[model_type]

    # Load the current model and check how long since it's been trained
    n_iteration, current_model = models.get_last_model()
    if current_model is None:
        # If no model has been trained so far, train a new one
        train_new_model(model_class, 0)
        return None

    last_training_current_model = current_model.end_dev
    today = pd.Timestamp.utcnow()
    hours_since_last_training = (today - last_training_current_model).total_seconds() // 3600

    # Train a new model every seven days
    if hours_since_last_training >= 24 * 7 or force:
        train_new_model(model_class, n_iteration + 1)
    else:
        logger.info(
            f"The current model was trained up to about {int(hours_since_last_training//24)} day(s) ago ({hours_since_last_training:.0f}h), no retraining necessary..."
        )


def make_plot(out_prediction: dict) -> None:
    """Make a plot with the latest predictions."""
    fig = PredictionFigure(out_prediction)
    fig.make_plot()
    fig.write_to_file()


parser = argparse.ArgumentParser(description="Prediction of energy demand in France")
subparsers = parser.add_subparsers(dest="subcommand")


# Download subcommand
download_parser = subparsers.add_parser("download")
download_parser.add_argument(
    "--force",
    action="store_true",
    dest="force",
    help="Force downloading of the data",
)
download_parser.add_argument(
    "dates",
    type=str,
    nargs="*",
    help="Start and end dates of the period to consider in the format YYYYMMDDHHmm. If no end date is provided, defaults to today. If no starting date is provided, defaults to the last recorded.",
)
# Prediction subcommand
predict_parser = subparsers.add_parser("predict")
# Training subcommand
train_parser = subparsers.add_parser("train")
train_parser.add_argument("model", type=str, nargs="?", help="Model to use: 'lgbm', 'xgb'")
train_parser.add_argument(
    "--force",
    action="store_true",
    dest="force",
    help="Force re-training of the data",
)
merge_parser = subparsers.add_parser("merge")
predict_parser.add_argument("--plot", action="store_true", dest="plot", help="Plot the prediction using the template.")

if __name__ == "__main__":
    stream_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    stream_handler.setLevel(logging.DEBUG)

    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(funcName)s(%(lineno)d) %(message)s")
    file_handler = RotatingFileHandler(config.path_log, maxBytes=2_000)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    logger = logging.getLogger(config.logger_name)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)

    dict_env = dotenv_values()
    args = parser.parse_args()
    if args.subcommand == "download":
        API_KEY = dict_env.get("API_KEY", "MISSING") or "MISSING"
        if len(args.dates) >= 2:
            download_new_data(api_key=API_KEY, start=args.dates[0], end=args.dates[1], force=args.force)
        elif len(args.dates) == 1:
            download_new_data(api_key=API_KEY, start=args.dates[0], force=args.force)
        else:
            download_new_data(api_key=API_KEY, force=args.force)
    if args.subcommand == "train":
        handle_training(model_type=args.model, force=args.force)
    if args.subcommand == "predict":
        out = get_model_prediction()
        if out is not None:
            logger.info(f"Forecast with metrics {out['metrics_future']}!")
            logger.info(f"Entsoe forecast with metrics {out['metrics_forecast']}!")
            if args.plot:
                make_plot(out)
    if args.subcommand == "merge":
        merge_build_manual()
