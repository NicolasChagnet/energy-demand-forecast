from src import config, data, models
from src.logger import logger
import datetime
from dotenv import dotenv_values
import argparse


def download_new_data(API_KEY):
    logger.info("Downloading new data...")
    current_data = data.load_energy()
    latest_idx = current_data.index[-1]
    today = datetime.datetime.now(tz=datetime.UTC)
    if ((today - latest_idx).total_seconds() // 3600) < 24:
        logger.warning("Last download less than 24h ago, aborting download...")
        return None
    _ = data.download_data(latest_idx + datetime.timedelta(hours=1), today, API_KEY)
    _ = data.merge_raw_files()
    _ = data.build_final_files()


def get_model_prediction(can_train_new=False):
    n_iteration, old_model = models.get_last_model()
    if (n_iteration < 0 or old_model is None or old_model.latest_mape > config.cutoff_mape) and can_train_new:
        logger.info("Making predictions using new model...")
        train_new_model()
    else:
        logger.info("Making predictions using previous model...")
        model = old_model
    y_future, y_future_pred = model.predict()
    mape_train, (y_train, y_train_pred) = model.get_training()
    return {
        "future_actual": y_future,
        "future_pred": y_future_pred,
        "train_actual": y_train,
        "train_pred": y_train_pred,
        "mape_future": model.latest_mape,
        "mape_train": mape_train,
    }


def train_new_model():
    logger.info("Training new model...")
    n_iteration, _ = models.get_last_model()
    current_data = data.load_energy()
    latest_idx = current_data.index[-1]
    model = models.ForecasterLGBM(n_iteration + 1, end_train=latest_idx - datetime.timedelta(days=1))
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

if __name__ == "__main__":
    dict_env = dotenv_values()
    args = parser.parse_args()
    if args.download:
        download_new_data(dict_env["API_KEY"])
    if args.train and not args.predict:
        train_new_model()
    if args.predict:
        (y, y_pred), mape = get_model_prediction(can_train_new=args.train)
        print(f"Forecast with MAPE {mape}!")
