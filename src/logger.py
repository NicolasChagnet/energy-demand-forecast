import logging
from src import config

logger = logging
logger.basicConfig(
    # filename=config.path_log, level=logging.INFO, filemode="a", format="%(asctime)s %(levelname)s %(message)s"
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
