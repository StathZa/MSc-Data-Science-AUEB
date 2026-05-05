from utils.libs import random, torch, DataLoader, Dataset

class BPRDataset(Dataset):
    """Each sample returns (user_idx, pos_item_idx, neg_item_idx)."""

    def __init__(self, df, user2idx, item2idx, user_pos_items, num_items):
        self.users = df['user_id'].map(user2idx).values
        self.items = df['item_id'].map(item2idx).values
        self.user_ids = df['user_id'].values
        self.user_pos = user_pos_items
        self.all_item_ids = list(item2idx.keys())
        self.item2idx = item2idx
        self.num_items = num_items

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        u = self.users[idx]
        pos = self.items[idx]
        uid = self.user_ids[idx]
        while True:
            neg_id = random.choice(self.all_item_ids)
            if neg_id not in self.user_pos[uid]:
                break
        neg = self.item2idx[neg_id]
        return torch.tensor(u), torch.tensor(pos), torch.tensor(neg)


def get_dataloader(df, user2idx, item2idx, user_pos_items, num_items,
                   batch_size=1024, shuffle=True):
    dataset = BPRDataset(df, user2idx, item2idx, user_pos_items, num_items)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)