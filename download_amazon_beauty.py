#!/usr/bin/env python3
"""
Quick script to download and preprocess Amazon Beauty dataset using RecBole.
This uses the minimal download-only configuration for fastest execution.

Usage:
    python download_amazon_beauty.py
    
Output (automatically saved by RecBole):
    - dataset/amazon-beauty/amazon-beauty.inter (all interactions)
    - dataset/amazon-beauty/amazon-beauty.item (item features)
    
Output (explicitly saved by this script):
    - dataset/amazon-beauty/amazon-beauty-train.inter
    - dataset/amazon-beauty/amazon-beauty-valid.inter
    - dataset/amazon-beauty/amazon-beauty-test.inter
"""

import os
import pandas as pd
import torch
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed

# Disable distributed processing for single-process use
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

"""Save interaction features to tab-separated file."""
def save_interactions(inter_feat, filename):
    data_dict = {}
    for field in inter_feat.interaction:
        field_data = inter_feat[field]
        # Convert torch tensors to numpy arrays
        if isinstance(field_data, torch.Tensor):
            data_dict[field] = field_data.cpu().numpy()
        else:
            data_dict[field] = field_data
    df = pd.DataFrame(data_dict)
    df.to_csv(filename, sep='\t', index=False)
    return len(df)

def main():
    print("Downloading and preprocessing Amazon Beauty dataset")
    
    config = Config(
        model='BPR',
        dataset='amazon-beauty',
        config_file_list=['amazon_beauty_download_only.yaml']
    )
    
    # Initialize random seed for reproducibility
    init_seed(config['seed'], config['reproducibility'])
    
    # Create dataset (triggers auto-download if needed)
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    print(f"Dataset: {len(dataset):,} interactions, {dataset.user_num:,} users, {dataset.item_num:,} items")
    print(f"Splits: Train={len(train_data.dataset):,}, Valid={len(valid_data.dataset):,}, Test={len(test_data.dataset):,}")
    
    # Save interaction splits (RecBole already saved .inter and .item files)
    base_path = 'dataset/amazon-beauty/amazon-beauty'
    save_interactions(train_data.dataset.inter_feat, f'{base_path}-train.inter')
    save_interactions(valid_data.dataset.inter_feat, f'{base_path}-valid.inter')
    save_interactions(test_data.dataset.inter_feat, f'{base_path}-test.inter')
    
    # Verify all files exist (RecBole automatically saves .inter and .item)
    item_file = f'{base_path}.item'
    inter_file = f'{base_path}.inter'
    
    if os.path.exists(item_file):
        size_mb = os.path.getsize(item_file) / (1024*1024)
        print(f"Item dataset: {item_file} ({size_mb:.1f} MB)")
    if os.path.exists(inter_file):
        size_mb = os.path.getsize(inter_file) / (1024*1024)
        print(f"Full interactions: {inter_file} ({size_mb:.1f} MB)")
    
    print("All files saved to dataset/amazon-beauty/")

if __name__ == '__main__':
    main()

