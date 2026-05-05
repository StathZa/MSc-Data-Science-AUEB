from utils.libs import np, pd, torch, logging, defaultdict, SentenceTransformer

def build_text_embeddings(items_df: pd.DataFrame, 
                          device: torch.device, 
                          model_name: str = "all-MiniLM-L6-v2", 
                          logger: logging.Logger = None):
  """Encode item text features with a frozen sentence transformer"""
  logger.info("Encode item features")
  st_model = SentenceTransformer(model_name, device=device)

  texts = (items_df["title"].fillna("") + " " + items_df["description"].fillna("")).tolist()
  texts = [t.strip()[:512] for t in texts]

  embeddings = st_model.encode(texts, batch_size=256, show_progress_bar=True)
  text_embed = torch.tensor(embeddings, dtype=torch.float32)

  logger.info(f"Text embedding shape: {text_embed.shape}")
  return text_embed

def build_numerical_features(items_df: pd.DataFrame, 
                            item2idx: dict,
                            logger: logging.Logger = None):
  """Normaliser of numerical features"""
  
  # num_cols = items_df.select_dtypes(["float", "int"]).columns.tolist()
  logger.info("Normalise numerical features for 2 Tower ingestion")

  items_df["log_price"] = np.log1p(items_df["price"])
  num_cols = ["log_price", "average_rating"]
  num_items = len(item2idx) 

  item_num = torch.zeros(num_items, len(num_cols), dtype=torch.float32)

  for i, col in enumerate(num_cols):
    mn, mx = items_df[col].min(), items_df[col].max()
    if mx > mn:
        items_df[col + '_norm'] = (items_df[col] - mn) / (mx - mn)
    else:
        items_df[col + '_norm'] = 0.0

    for row in items_df.itertuples():
        idx = item2idx[row.item_id]
        item_num[idx, i] = getattr(row, col + '_norm')

    logger.info(f"Numerical features shape: {item_num.shape}")
    return item_num

def build_categorical_features(items_df: pd.DataFrame, 
                               item2idx: dict, 
                               cat_vocab: dict,
                               logger: logging.Logger = None):
    """Map categorical features to indices."""
    num_items = len(item2idx)
    item_cat = torch.zeros(num_items, 2, dtype=torch.long)

    for row in items_df.itertuples():
        idx = item2idx[row.item_id]
        item_cat[idx, 0] = cat_vocab['main_category'].get(row.main_category, 0)
        item_cat[idx, 1] = cat_vocab['store'].get(row.store, 0)

    logger.info(f"Categorical features shape: {item_cat.shape}")
    return item_cat
