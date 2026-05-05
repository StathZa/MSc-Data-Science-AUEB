"""Data loading, vocabulary construction, and train/test splitting."""
# Import Dependencies
from datasets import load_dataset
from utils.config import *
import logging

# Download & load
def _stream_to_dataframe(parameters: dict, logger, max_rows: int) -> pd.DataFrame:
    """Stream dataset and return first max_rows as a DataFrame."""
    stream_params = parameters.copy()
    stream_params["streaming"] = True
    ds = load_dataset(**stream_params)
    rows = []
    for i, row in enumerate(ds):
        if i >= max_rows:
            break
        rows.append(row)
    logger.info(f"Streamed {len(rows):,} rows")
    return pd.DataFrame(rows)


def fetch_review_data(
    parameters: dict,
    logger: logging.Logger = None,
    folder_name: str = "reviews",
    file_name: str = "review_data.parquet.gzip",
    max_rows: int = 500_000
) -> pd.DataFrame:
    """Download review data from HuggingFace or load from local cache.

    On first call, streams the dataset via load_dataset() and saves the
    first max_rows as a compressed parquet file under ./data/{folder_name}/.
    On subsequent calls, reads directly from the cached file.

    Args:
        parameters: Keyword arguments passed to datasets.load_dataset().
            Expected keys: path, name, split, trust_remote_code, token.
        logger: Logger instance for tracking execution and errors.
        folder_name: Subdirectory name under ./data/ for storing the file.
        file_name: Name of the parquet file to save/load.
        max_rows: Maximum number of rows to stream from the dataset.

    Returns:
        DataFrame containing the review data. Returns an empty DataFrame
        if the download fails.
    """
    # Check whether data exist
    if not data_dir.exists():
        logger.warning(f"Creating data directory to persist the reviews and metadata content")
        data_dir.mkdir(parents=True)
        logger.info(f"Data directory {data_dir} created. Downloading data from source")
        try:
            rev_df = _stream_to_dataframe(parameters, logger, max_rows)
            rev_df.to_parquet(file, compression="gzip")
        except Exception as e:
            logger.exception(f"Could not load the data since {e} emerged.")
            rev_df = pd.DataFrame()
    else:
        if len(os.listdir(data_dir)) == 0:
            logger.warning(f"Data directory is empty. Downloading data from source...")
            try:
                rev_df = _stream_to_dataframe(parameters, logger, max_rows)
                rev_df.to_parquet(file, compression="gzip")
            except Exception as e:
                logger.exception(f"Could not load the data since {e} emerged.")
                rev_df = pd.DataFrame()
        else:
            logger.info("Data directory already exists and contains data...\nImporting data from local filesystem's location")
            rev_df = pd.read_parquet(file)

    return rev_df


def fetch_review_metadata(
    parameters: dict,
    logger: logging.Logger = None,
    folder_name: str = "metadata",
    file_name: str = "review_metadata.parquet.gzip",
    max_rows: int = 500_000
) -> pd.DataFrame:
    """Download item metadata from HuggingFace or load from local cache.

    Args:
        parameters: Keyword arguments passed to datasets.load_dataset().
            Expected keys: path, name, split, trust_remote_code, token.
        logger: Logger instance for tracking execution and errors.
        folder_name: Subdirectory name under ./data/ for storing the file.
        file_name: Name of the parquet file to save/load.
        max_rows: Maximum number of rows to stream from the dataset.

    Returns:
        DataFrame containing the item metadata. Returns an empty DataFrame
        if the download fails.
    """

    # Check whether data exist
    if not data_dir.exists():
        logger.warning(f"Creating data directory to persist the reviews and metadata content")
        data_dir.mkdir(parents=True)
        logger.info(f"Data directory {data_dir} created. Downloading data from source")
        try:
            meta_df = _stream_to_dataframe(parameters, logger, max_rows) 
            meta_df.to_parquet(file, compression="gzip")
        except Exception as e:
            logger.exception(f"Could not load the data since {e} emerged.")
            meta_df = pd.DataFrame()
    else:
        if len(os.listdir(data_dir)) == 0:
            logger.warning(f"Data directory is empty. Downloading data from source...")
            try:
                meta_df = _stream_to_dataframe(parameters, logger, max_rows) 
                meta_df.to_parquet(file, compression="gzip")
            except Exception as e:
                logger.exception(f"Could not load the data since {e} emerged.")
                meta_df = pd.DataFrame()
        else:
            logger.info("Data directory already exists and contains data...\nImporting data from local filesystem's location")
            meta_df = pd.read_parquet(file)

    return meta_df
 