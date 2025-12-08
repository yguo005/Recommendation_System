"""
Utility functions for consistent random-negative evaluation across models.

all moedels (baselines, Two-Tower, DeepFM, XGBoost) should import from this
file so that:
  - sample negatives in exactly the same way
  - compute recall@K / ndcg@K / num_users_eval with the same logic
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
import sys

"""
    Build an evaluation dataframe with 1 positive + `num_neg` random negatives per user.
    
    Args
   
    train_df : pd.DataFrame
        the train split (must include user_id, item_id, label).
    test_df : pd.DataFrame
        the test split (must include user_id, item_id, label).
    num_neg : int, default 5
        Number of random negative items per user.
    seed : int, default 42
        RNG seed for reproducibility.
    max_users : int, optional
        Limit number of users for faster evaluation (for testing).

    Returns
   
    eval_df : pd.DataFrame
        DataFrame with columns [user_id, item_id, label].
"""
def build_random_neg_eval_df(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_neg: int = 5,
    seed: int = 42,
    max_users: int = None,
) -> pd.DataFrame:
    
    rng = np.random.default_rng(seed)

    print("  Filtering positives")
    sys.stdout.flush()
    
    # Work only with positives
    train_pos = train_df[train_df["label"] == 1].copy()
    test_pos = test_df[test_df["label"] == 1].copy()

    if train_pos.empty or test_pos.empty:
        raise ValueError("train_df or test_df has no positive (label==1) rows.")

    # Universe of items that appear positively anywhere
    all_pos_items = np.unique(
        np.concatenate([train_pos["item_id"].values, test_pos["item_id"].values])
    )
    all_pos_items_set = set(all_pos_items)
    n_items = len(all_pos_items)
    print(f"  Item universe: {n_items:,} items")
    sys.stdout.flush()

    # Get unique test users (one positive per user)
    print("  Getting unique test users")
    sys.stdout.flush()
    test_user_pos = test_pos.groupby("user_id")["item_id"].first().reset_index()
    test_user_pos.columns = ["user_id", "pos_item"]
    
    if max_users and len(test_user_pos) > max_users:
        test_user_pos = test_user_pos.sample(n=max_users, random_state=seed)
    
    n_users = len(test_user_pos)
    print(f"  Evaluating {n_users:,} users with {num_neg} negatives each")
    sys.stdout.flush()

    # Build user -> seen items mapping (for excluding from negatives)
    print("  Building seen-items index")
    sys.stdout.flush()
    all_pos = pd.concat([train_pos, test_pos], ignore_index=True)
    user_seen = all_pos.groupby("user_id")["item_id"].apply(set).to_dict()

    # Generate rows
    print("  Sampling negatives")
    sys.stdout.flush()
    
    rows = []
    progress_step = max(1, n_users // 10)
    
    for i, (user, pos_item) in enumerate(zip(test_user_pos["user_id"], test_user_pos["pos_item"])):
        if i % progress_step == 0:
            pct = 100 * i / n_users
            print(f"    Progress: {i:,}/{n_users:,} ({pct:.0f}%)")
            sys.stdout.flush()
        
        # Add positive
        rows.append({"user_id": user, "item_id": pos_item, "label": 1})
        
        # Sample negatives (exclude user's seen items)
        seen = user_seen.get(user, set())
        
        # sample more than needed, filter, take first num_neg
        n_sample = min(num_neg * 3, n_items)
        candidates = rng.choice(all_pos_items, size=n_sample, replace=False)
        neg_items = [c for c in candidates if c not in seen][:num_neg]
        
        # Fallback if not enough
        if len(neg_items) < num_neg:
            neg_items = list(rng.choice(all_pos_items, size=num_neg, replace=False))
        
        for ni in neg_items:
            rows.append({"user_id": user, "item_id": ni, "label": 0})

    print(f"    Progress: {n_users:,}/{n_users:,} (100%)")
    sys.stdout.flush()
    
    eval_df = pd.DataFrame(rows)
    print(f"  Built eval_df: {len(eval_df):,} rows")
    sys.stdout.flush()
    return eval_df

"""
    Compute recall@K and ndcg@K for a model given scores on a random-negative eval_df.

    Assumes:
      - eval_df contains columns: user_id, item_id, label (1 or 0), and `score_col`.
      - For each user, eval_df has exactly one positive (label==1) and `num_neg` negatives.

    Metrics:
      - recall@K: fraction of users where the positive is ranked in top-K.
      - ndcg@K:   DCG@K with a single relevant item / ideal DCG (which is 1.0).
      - num_users: number of users actually evaluated.

    Returns a dict including:
      - "recall@10", "recall@20", ...
      - "ndcg@10", "ndcg@20", ...
      - "num_users"
"""
def evaluate_recall_ndcg_at_k(
    eval_df: pd.DataFrame,
    score_col: str,
    ks: Tuple[int, ...] = (10, 20),
) -> Dict[str, float]:
    
    metrics: Dict[str, float] = {}
    total_users = 0

    for k in ks:
        metrics[f"recall@{k}"] = 0.0
        metrics[f"ndcg@{k}"] = 0.0

    # Vectorized ranking per user
    eval_df = eval_df.copy()
    eval_df["rank"] = eval_df.groupby("user_id")[score_col].rank(ascending=False, method="first")
    
    # Get positive rows only
    pos_df = eval_df[eval_df["label"] == 1]
    total_users = len(pos_df)
    
    if total_users == 0:
        metrics["num_users"] = 0
        return metrics
    
    for k in ks:
        # Count how many positives are ranked <= k
        hits = (pos_df["rank"] <= k).sum()
        metrics[f"recall@{k}"] = hits / total_users
        
        # NDCG: 1/log2(rank+1) for hits in top-k
        in_topk = pos_df[pos_df["rank"] <= k]
        ndcg_sum = (1.0 / np.log2(in_topk["rank"] + 1)).sum()
        metrics[f"ndcg@{k}"] = ndcg_sum / total_users

    metrics["num_users"] = float(total_users)

    return metrics


__all__ = [
    "build_random_neg_eval_df",
    "evaluate_recall_ndcg_at_k",
]
