"""
Data Loading for EuroSAT Dataset
Handles download, splitting, and DataLoader creation.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
import os
import argparse

class EuroSATConfig:
    DATA_PATH = './data'
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    NUM_CLASSES = 10
    CLASS_NAMES = [
        'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
        'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
        'River', 'SeaLake'
    ]

def load_eurosat_dataset(download=False):
    """
    Load and split EuroSAT dataset.
    
    Args:
        download (bool): Whether to download if not present.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    full_dataset = datasets.EuroSAT(
        root=EuroSATConfig.DATA_PATH, download=download,
        transform=None  # Transforms applied later
    )
    
    total_size = len(full_dataset)
    train_size = int(EuroSATConfig.TRAIN_RATIO * total_size)
    val_size = int(EuroSATConfig.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(EuroSATConfig.RANDOM_SEED)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    print(f"Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    return train_dataset, val_dataset, test_dataset

def create_loaders(train_dataset, val_dataset, test_dataset, train_transform=None, test_transform=None):
    """
    Create DataLoaders with optional transforms.
    """
    if train_transform:
        train_dataset.dataset.transform = train_transform
    if test_transform:
        val_dataset.dataset.transform = test_transform
        test_dataset.dataset.transform = test_transform
    
    train_loader = DataLoader(
        train_dataset, batch_size=EuroSATConfig.BATCH_SIZE,
        shuffle=True, num_workers=EuroSATConfig.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=EuroSATConfig.BATCH_SIZE,
        shuffle=False, num_workers=EuroSATConfig.NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=EuroSATConfig.BATCH_SIZE,
        shuffle=False, num_workers=EuroSATConfig.NUM_WORKERS, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='Download dataset')
    args = parser.parse_args()
    
    train_ds, val_ds, test_ds = load_eurosat_dataset(download=args.download)
    print("Dataset loaded successfully!")
