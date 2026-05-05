from utils.libs import pd, defaultdict, logging

def train_val_test_split(df: pd.DataFrame, 
                     user2idx: dict, 
                     item2idx: dict,
                     logger: logging.Logger = None):
    """Leave-two-out split: last interaction for test, second-to-last for validation"""
    df = df.sort_values('timestamp')

    user_pos_items = defaultdict(set)
    for row in df.itertuples():
        user_pos_items[row.user_id].add(row.item_id)

    train_rows, val_rows, test_rows = [], [], []
    for uid, group in df.groupby('user_id'):
        items = group.index.tolist()
        if len(items) < 3:
            train_rows.extend(items)
            continue
        train_rows.extend(items[:-2])
        val_rows.append(items[-2])
        test_rows.append(items[-1])

    train_df = df.loc[train_rows]
    val_df = df.loc[val_rows]
    test_df = df.loc[test_rows]

    logger.info(f"Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")
    return train_df, val_df, test_df, user_pos_items
