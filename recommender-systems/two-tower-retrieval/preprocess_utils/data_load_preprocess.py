"""Data loading, vocabulary construction, and train/test splitting."""
# Import Dependencies
from utils.config import *
from utils.libs import os, np, pd, logging, load_dataset, Path

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
    data_dir: str = Path(os.getcwd()) / "data" / folder_name
    file: str = str(data_dir / file_name)

    logger.info("Checking filesystem for the data")

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
    data_dir: str = Path(os.getcwd()) / "data" / folder_name
    file: str = str(data_dir / file_name)

    logger.info("Checking filesystem for the data")

    
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
 
def prepare_training_data(reviews: pd.DataFrame,                          
                    metadata: pd.DataFrame,
                    review_cols_to_keep: list = ['user_id', 'parent_asin', 'rating', 'timestamp'],
                    metadata_cols_to_keep: list = ['parent_asin', 'main_category', 'store', 'title',
                                                   'average_rating', 'rating_number', 'price', 'description'],
                    min_pos_rate: int = 4,
                    min_interactions: int = 2,
                    max_interactions: int = 200_000,
                    interactions_ratio: float = 0.1 ,                
                    logger: logging.Logger = None) -> dict:           
    """Preprocess reviews and metadata into a training-ready DataFrame.

    Performs three steps:
      1. Cleanup: filter positive interactions, fill missing values, merge datasets.
      2. Filtering: k-core filtering (users and items with >= min_interactions)
         and optional subsampling.
      3. Vocabulary: build user/item index mappings and categorical vocabularies.

    Args:
        reviews: Raw review DataFrame from fetch_review_data().
        metadata: Raw metadata DataFrame from fetch_review_metadata().
        review_cols_to_keep: Review columns to retain before merging.
        metadata_cols_to_keep: Metadata columns to retain before merging.
        min_pos_rate: Minimum rating to count as a positive interaction.
        min_interactions: Minimum interactions per user/item for k-core filtering.
        max_interactions: Maximum total interactions (subsample if exceeded).
        interactions_ratio: A float that represents the lowest allowed percentage of surviving interactions after K-filtering
        logger: Logger instance for tracking execution.

    Returns:
        Dictionary with keys:
          - 'df': Preprocessed DataFrame
          - 'user2idx': User to index mapping
          - 'item2idx': Item to index mapping
          - 'cat_vocab': Categorical vocabulary mappings
          - 'items_df': Deduplicated item-level DataFrame
    """
    logger.info("\nPreprocessing data")
    logger.info("=" * 50)
    logger.info("PART 1 - CLEANUP")
    logger.info("=" * 50)

    reviews = reviews[review_cols_to_keep].copy()
    reviews.rename(columns={"parent_asin": "item_id"}, inplace=True)

    logger.info(f"Trimming reviews with rating >= {min_pos_rate} as positive interactions.")
    reviews = reviews.loc[reviews.rating >= min_pos_rate].reset_index(drop=True).copy()

    meta = metadata[metadata_cols_to_keep].copy()
    meta.rename(columns={"parent_asin": "item_id"}, inplace=True)

    meta["price"] = pd.to_numeric(meta["price"], errors="coerce")

    logger.info("Fill in missing numerical data with median.")
    for col in meta.select_dtypes(include=[np.number]).columns:
        meta[col] = meta[col].fillna(meta[col].median())        

    meta["title"] = meta["title"].fillna("")
    meta["main_category"] = meta["main_category"].fillna("Unknown")
    meta["store"] = meta["store"].fillna("Unknown")
    meta["description"] = meta["description"].apply(
        lambda x: ' '.join(x) if isinstance(x, list) else (x if isinstance(x, str) else '')
    )

    logger.info(f"Positive interactions: {len(reviews):,}")
    logger.info(f"Items with metadata: {len(meta):,}")

    logger.info("Merging user interactions with item metadata...")
    df = reviews.merge(meta, on="item_id", how="inner")
    logger.info(f"Merged interactions: {len(df):,}")

    logger.info("=" * 50)
    logger.info("PART 2 - FILTERING & SUBSAMPLE")
    logger.info("=" * 50)

    # K-core filtering
    df_full = df.copy()
    for i in range(min_interactions):
        user_cnts = df["user_id"].value_counts()
        item_cnts = df["item_id"].value_counts()                     
        df = df[df["user_id"].isin(user_cnts[user_cnts >= min_interactions].index)]
        df = df[df["item_id"].isin(item_cnts[item_cnts >= min_interactions].index)]
    
    logger.info(f"After k-core filtering: {len(df):,} interactions, "
                f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")
    
    # Safety check to prevent removal of great amount of interactions 
    if df.shape[0] < interactions_ratio * df_full.shape[0]:
      logger.warning(f"K-core filtering dropped {100 * (1 - len(df)/len(df_full)):.1f}% of data. "
                     f"Falling back to merged data without filtering.")
      df = df_full

    # Subsample if too large
    if len(df) > max_interactions:
        logger.info(f"Subsampling from {len(df):,} to ~{max_interactions:,} interactions")
        sampled_users = df['user_id'].drop_duplicates().sample(
            frac=max_interactions / len(df), random_state=SEED
        )
        df = df[df['user_id'].isin(sampled_users)]

        # Re-run k-core after subsampling
        for _ in range(3):
            user_cnts = df['user_id'].value_counts()                
            item_cnts = df['item_id'].value_counts()                
            df = df[df['user_id'].isin(user_cnts[user_cnts >= min_interactions].index)]
            df = df[df['item_id'].isin(item_cnts[item_cnts >= min_interactions].index)]

    logger.info(f"Working set: {len(df):,} interactions, "
                f"{df['user_id'].nunique():,} users, {df['item_id'].nunique():,} items")

    logger.info("=" * 50)
    logger.info("PART 3 - BUILD VOCABULARY")
    logger.info("=" * 50)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    user_ids = sorted(df['user_id'].unique())
    item_ids = sorted(df['item_id'].unique())
    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {it: i for i, it in enumerate(item_ids)}

    logger.info(f"num_users={len(user2idx):,}, num_items={len(item2idx):,}")

    # Build item-level dataframe
    items_df = df.drop_duplicates('item_id').set_index('item_id').loc[item_ids].reset_index()

    cat_vocab = {}
    for col in ['main_category', 'store']:
        uniques = ['<UNK>'] + sorted(items_df[col].unique().tolist())
        cat_vocab[col] = {v: i for i, v in enumerate(uniques)}

    logger.info(f"main_category vocab size: {len(cat_vocab['main_category'])}")
    logger.info(f"store vocab size: {len(cat_vocab['store'])}")

    return {
        'df': df,
        'user2idx': user2idx,
        'item2idx': item2idx,
        'cat_vocab': cat_vocab,
        'items_df': items_df
    }