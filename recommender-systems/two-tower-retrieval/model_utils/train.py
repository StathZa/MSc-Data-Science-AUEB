from utils.libs import torch, F

def bpr_loss(user_emb, pos_emb, neg_emb):
    pos_scores = (user_emb * pos_emb).sum(dim=-1)
    neg_scores = (user_emb * neg_emb).sum(dim=-1)
    return -F.logsigmoid(pos_scores - neg_scores).mean()


def get_item_features(item_idx, item_cat, item_num, text_emb):
    """Retrieve precomputed item features for a batch of item indices."""
    return item_cat[item_idx], item_num[item_idx], text_emb[item_idx]


def train_one_epoch(model, train_loader, optimizer, item_cat, item_num, text_emb, device):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for user_idx, pos_idx, neg_idx in train_loader:
        user_idx = user_idx.to(device)
        pos_idx = pos_idx.to(device)
        neg_idx = neg_idx.to(device)

        pos_cat, pos_num, pos_text = get_item_features(pos_idx, item_cat, item_num, text_emb)
        neg_cat, neg_num, neg_text = get_item_features(neg_idx, item_cat, item_num, text_emb)

        user_emb, pos_emb = model(user_idx, pos_cat, pos_num, pos_text)
        _, neg_emb = model(user_idx, neg_cat, neg_num, neg_text)

        loss = bpr_loss(user_emb, pos_emb, neg_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * user_idx.size(0)
        n_samples += user_idx.size(0)

    return total_loss / n_samples