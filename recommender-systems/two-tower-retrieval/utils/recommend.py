from utils.libs import torch, F

@torch.no_grad()
def recommend_for_user(model, user_id, user2idx, item2idx, items_df,
                       item_cat, item_num, text_emb, device,
                       user_pos_items=None, k=10):
    """Return top-k recommended items for a given user."""
    model.eval()
    idx2item = {v: k for k, v in item2idx.items()}

    u_t = torch.tensor([user2idx[user_id]], device=device)
    u_emb = F.normalize(model.user_tower(u_t), dim=-1)

    all_idx = torch.arange(len(item2idx), device=device)
    cat_f, num_f, txt_f = item_cat[all_idx], item_num[all_idx], text_emb[all_idx]
    all_item_emb = F.normalize(model.item_tower(cat_f, num_f, txt_f), dim=-1)

    scores = (u_emb * all_item_emb).sum(dim=-1)

    # Optionally mask already-seen items
    if user_pos_items and user_id in user_pos_items:
        for seen_item in user_pos_items[user_id]:
            if seen_item in item2idx:
                scores[item2idx[seen_item]] = -1.0

    topk = scores.topk(k)
    results = []
    for idx, score in zip(topk.indices.cpu().numpy(), topk.values.cpu().numpy()):
        iid = idx2item[idx]
        row = items_df.loc[items_df['item_id'] == iid].iloc[0]
        results.append({
            'item_id': iid,
            'title': row['title'][:80],
            'category': row['main_category'],
            'price': row['price'],
            'score': float(score)
        })
    return results