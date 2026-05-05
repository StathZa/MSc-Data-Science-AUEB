from utils.libs import math, random, torch, F

def get_item_features(item_idx, item_cat, item_num, text_emb):
    return item_cat[item_idx], item_num[item_idx], text_emb[item_idx]


@torch.no_grad()
def evaluate(model, test_df, user2idx, item2idx, user_pos_items,
             item_cat, item_num, text_emb, device,
             num_neg=99, K_list=[5, 10, 20], logger=None):
    """Evaluate with Hit Rate@K and NDCG@K against random negatives."""
    model.eval()
    item_ids = list(item2idx.keys())

    hits = {k: 0 for k in K_list}
    ndcgs = {k: 0.0 for k in K_list}
    count = 0

    for row in test_df.itertuples():
        uid = row.user_id
        pos_item = row.item_id

        if uid not in user2idx or pos_item not in item2idx:
            continue

        # Sample negatives
        neg_items = []
        while len(neg_items) < num_neg:
            cand = random.choice(item_ids)
            if cand not in user_pos_items[uid] and cand != pos_item:
                neg_items.append(cand)

        cand_ids = [pos_item] + neg_items
        cand_idx = torch.tensor([item2idx[c] for c in cand_ids], device=device)
        user_t = torch.tensor([user2idx[uid]], device=device)

        cat_f, num_f, txt_f = get_item_features(cand_idx, item_cat, item_num, text_emb)
        user_emb = F.normalize(model.user_tower(user_t), dim=-1)
        item_emb = F.normalize(model.item_tower(cat_f, num_f, txt_f), dim=-1)

        scores = (user_emb * item_emb).sum(dim=-1)
        _, topk_idx = scores.topk(max(K_list))
        topk_idx = topk_idx.cpu().numpy()

        for k in K_list:
            top = topk_idx[:k]
            if 0 in top:  # index 0 = positive item
                hits[k] += 1
                rank = int((top == 0).argmax()) + 1
                ndcgs[k] += 1.0 / math.log2(rank + 1)
        count += 1

    results = {}
    for k in K_list:
        results[f'HR@{k}'] = hits[k] / count
        results[f'NDCG@{k}'] = ndcgs[k] / count

    if logger:
        for metric, val in results.items():
            logger.info(f"  {metric}: {val:.4f}")

    return results