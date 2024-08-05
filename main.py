from src import config, data, models
from src.logger import logger
import datetime
from dotenv import dotenv_values
import argparse
import pandas as pd


def merge_build_manual():
    _ = data.merge_raw_files()
    _ = data.build_final_files()


def download_new_data(API_KEY, force=False):
    logger.info("Downloading new data...")
    current_data = data.load_energy()
    latest_idx = current_data.index[-1]
    today = datetime.datetime.now(tz=datetime.timezone.utc)
    if ((today - latest_idx).total_seconds() // 3600) < 24 and not force:
        logger.warning("Last download less than 24h ago, aborting download...")
        return None
    _ = data.download_data(latest_idx + datetime.timedelta(hours=1), today, API_KEY)
    _ = data.merge_raw_files()
    _ = data.build_final_files()


def get_model_prediction():
    n_iteration, model = models.get_last_model()
    if n_iteration < 0 or model is None:
        logger.error("No model found, train a model first!")
        return None
    logger.info("Making predictions using previous model...")
    mape, (y_future, y_future_pred) = model.predict()
    mape_train, (y_train, y_train_pred) = model.get_training()
    return {
        "future_actual": y_future,
        "future_pred": y_future_pred,
        "train_actual": y_train,
        "train_pred": y_train_pred,
        "mape_future": mape,
        "mape_train": mape_train,
    }


def train_new_model(force=False):
    logger.info("Training new model...")
    n_iteration, old_model = models.get_last_model()
    current_data = data.load_energy()
    latest_idx = current_data.index[-1]
    today = datetime.datetime.now(tz=datetime.timezone.utc)
    more_week = ((today - latest_idx).total_seconds() // 3600) >= 24 * 7

    should_retrain = n_iteration < 0 or old_model is None or old_model.predict()[0] > config.cutoff_mape
    if should_retrain or force or more_week:
        logger.info("Retraining needed!")
        end_train_cutoff = (
            latest_idx - datetime.timedelta(days=1)
            if n_iteration >= 0
            else pd.to_datetime(config.end_train_default, utc=True)
        )
        model = models.ForecasterLGBM(n_iteration + 1, end_train=end_train_cutoff)
        model.tune()


parser = argparse.ArgumentParser(description="Prediction of energy demand in France")
parser.add_argument(
    "-d",
    "--download",
    action="store_true",
    dest="download",
    help="Download new data",
)
parser.add_argument(
    "-p",
    "--predict",
    action="store_true",
    dest="predict",
    help="Predict data",
)
parser.add_argument(
    "-t",
    "--train",
    action="store_true",
    dest="train",
    help="Train and tune a new model",
)
parser.add_argument(
    "--force",
    action="store_true",
    dest="force",
    help="Force training of new model",
)
parser.add_argument(
    "--merge",
    action="store_true",
    dest="merge",
    help="Merge raw data into final build files",
)

if __name__ == "__main__":
    dict_env = dotenv_values()
    args = parser.parse_args()
    if args.download:
        download_new_data(dict_env["API_KEY"], force=args.force)
    if args.train and not args.predict:
        train_new_model(force=args.force)
    if args.predict:
        out = get_model_prediction()
        print(f"Forecast with MAPE {out['mape_future']}!")
    if args.merge:
        merge_build_manual()
