from model_utils.model import TwoTowerModel
from model_utils.train import train_one_epoch
from evaluate_utils.evaluate import evaluate
from preprocess_utils.dataset import get_dataloader
from utils.libs import torch, nn, product
from utils.config import param_grid, MAX_EPOCHS, PATIENCE

def hyperparameter_search(train_df, val_df, user2idx, item2idx, cat_vocab,
                          user_pos_items, item_cat, item_num, text_emb,
                          device, logger=None):
    """Grid search over key hyperparameters, evaluated on validation set."""

    cat_vocab_sizes = [len(cat_vocab['main_category']), len(cat_vocab['store'])]
    text_dim = text_emb.shape[1]
    num_users = len(user2idx)
    num_items = len(item2idx)

    all_results = []
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    logger.info(f"Running grid search: {len(combos)} combinations")

    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        logger.info(f"\n[{i+1}/{len(combos)}] {params}")

        loader = get_dataloader(train_df, user2idx, item2idx, user_pos_items,
                                num_items, batch_size=params['batch_size'])

        model = TwoTowerModel(
            num_users=num_users,
            cat_vocab_sizes=cat_vocab_sizes,
            cat_emb_dim=16,
            num_numeric=2,
            text_dim=text_dim,
            embed_dim=params['embed_dim']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

        best_ndcg = 0.0
        patience_counter = 0
        best_epoch = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            loss = train_one_epoch(model, loader, optimizer,
                                   item_cat, item_num, text_emb, device)
            scheduler.step()

            val_res = evaluate(model, val_df, user2idx, item2idx, user_pos_items,
                               item_cat, item_num, text_emb, device, logger=None)

            if val_res['NDCG@10'] > best_ndcg:
                best_ndcg = val_res['NDCG@10']
                best_epoch = epoch
                best_val = val_res.copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        logger.info(f"  Best epoch {best_epoch}: NDCG@10={best_ndcg:.4f}")
        all_results.append({
            **params,
            'best_epoch': best_epoch,
            **{f'best_{k}': v for k, v in best_val.items()}
        })

    return all_results