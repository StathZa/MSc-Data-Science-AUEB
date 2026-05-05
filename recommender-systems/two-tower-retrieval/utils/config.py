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

reviews_params = {"path": reviews_path,
                  "name": reviews_name,
                  "split": "full",}
metadata_params = {"path": reviews_path, 
                   "name": metadata_name,
                   "split": "full",}


