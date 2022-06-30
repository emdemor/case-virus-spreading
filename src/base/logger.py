import os
import logging
import yaml
from datetime import datetime


with open("config/filepaths.yaml", "r") as file:
    filepaths = yaml.safe_load(file)

filename = os.path.join(
    filepaths["logs_directory_path"],
    "history.log"
    # datetime.strftime(datetime.now(), "%Y-%m-%d__%H_%M_%s.log"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(filename), logging.StreamHandler()],
)
