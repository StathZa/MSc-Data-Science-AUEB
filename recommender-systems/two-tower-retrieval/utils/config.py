# Parameter Configuration file
from utils.libs import os, Path

# Reproducible seed 
SEED: int = 42

USE_COLAB: bool = False

# Remote directories and configuration
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

# Local filenames
reviews_file = "review_data.parquet.gzip"
metadata_file = "review_metadata.parquet.gzip"

# Embeddings
EMBED_DIM = 64   # final embedding dimension for both towers
CAT_EMB   = 16   # dimension for each categorical embedding

# Training layout
EPOCHS = 50
PATIENCE = 7
MAX_EPOCHS = 70

# hyperparameter tuning
param_grid = {
    'embed_dim': [32, 64, 128],
    'lr': [1e-2, 1e-3, 5e-4],
    'batch_size': [512, 1024],
}