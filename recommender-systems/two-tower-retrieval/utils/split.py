from utils.libs import pd, defaultdict, logging

def train_test_split(df: pd.DataFrame, 
                     user2idx: dict, 
                     item2idx: dict,
                     logger: logging.Logger = None):
  """Leave-one-out split: last interaction per user goes to test."""
  df = df.sort_values('timestamp')

  user_pos_items = defaultdict(set)
  for row in df.itertuples():
      user_pos_items[row.user_id].add(row.item_id)

  train_rows, test_rows = [], []
  for uid, group in df.groupby('user_id'):
      items = group['item_id'].tolist()
      if len(items) < 2:
          train_rows.extend(group.index.tolist())
          continue
      train_rows.extend(group.iloc[:-1].index.tolist())
      test_rows.append(group.iloc[-1].name)

  train_df = df.loc[train_rows]
  test_df = df.loc[test_rows]

  logger.info(f"Train: {len(train_df):,}  |  Test: {len(test_df):,}")
  return train_df, test_df, user_pos_items
