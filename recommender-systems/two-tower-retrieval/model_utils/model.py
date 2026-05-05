from utils.libs import torch, nn, F
from utils.config import EMBED_DIM, CAT_EMB


class UserTower(nn.Module):
    def __init__(self, num_users, embed_dim):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

    def forward(self, user_idx):
        return self.user_emb(user_idx)


class ItemTower(nn.Module):
    """Aggregates categorical, numerical, and text features."""

    def __init__(self, cat_vocab_sizes, cat_emb_dim, num_numeric, text_dim, embed_dim):
        super().__init__()
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vs, cat_emb_dim) for vs in cat_vocab_sizes
        ])
        for emb in self.cat_embeddings:
            nn.init.xavier_uniform_(emb.weight)

        self.num_proj = nn.Linear(num_numeric, cat_emb_dim)

        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128),
            nn.ReLU(),
            nn.Linear(128, cat_emb_dim)
        )

        total_in = cat_emb_dim * (len(cat_vocab_sizes) + 2)  # cats + num + text
        self.fusion = nn.Sequential(
            nn.Linear(total_in, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, embed_dim)
        )

    def forward(self, cat_feats, num_feats, text_feats):
        parts = []
        for i, emb_layer in enumerate(self.cat_embeddings):
            parts.append(emb_layer(cat_feats[:, i]))
        parts.append(self.num_proj(num_feats))
        parts.append(self.text_proj(text_feats))
        x = torch.cat(parts, dim=-1)
        return self.fusion(x)


class TwoTowerModel(nn.Module):
    def __init__(self, num_users, cat_vocab_sizes, cat_emb_dim,
                 num_numeric, text_dim, embed_dim):
        super().__init__()
        self.user_tower = UserTower(num_users, embed_dim)
        self.item_tower = ItemTower(cat_vocab_sizes, cat_emb_dim,
                                    num_numeric, text_dim, embed_dim)

    def forward(self, user_idx, cat_feats, num_feats, text_feats):
        user_emb = self.user_tower(user_idx)
        item_emb = self.item_tower(cat_feats, num_feats, text_feats)
        user_emb = F.normalize(user_emb, dim=-1)
        item_emb = F.normalize(item_emb, dim=-1)
        return user_emb, item_emb