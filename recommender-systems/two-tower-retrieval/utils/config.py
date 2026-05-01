# Parameter Configuration file
import os
from pathlib import Path

# Reproducible seed 
SEED: int = 42

USE_COLAB: bool = False

reviews_path = "McAuley-Lab/Amazon-Reviews-2023"
reviews_name = "raw_review_Electronics"
metadata_name = "raw_meta_Electronics"

data_directory = Path(os.getcwd()) / 'data'

reviews_params = {"path": reviews_path, "name": reviews_name, "data_dir": data_directory,
                  "split": "full","revision": "refs/convert/parquet"}
metadata_params = {"path": reviews_path, "name": metadata_name, "data_dir": data_directory,
                   "split": "full", "revision": "refs/convert/parquet"}


