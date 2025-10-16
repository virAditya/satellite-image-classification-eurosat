"""
Data Augmentation for EuroSAT
Custom transforms for satellite images.
"""

from torchvision import transforms

def get_train_transform():
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),  # Satellite images can be rotated
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Atmospheric effects
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_test_transform():
    """Test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# For TTA (Test-Time Augmentation)
def get_tta_transforms():
    """Multiple transforms for TTA averaging."""
    return [
        get_test_transform(),
        transforms.Compose([
            transforms.RandomHorizontalFlip(p=1.0),
            get_test_transform()
        ]),
        transforms.Compose([
            transforms.RandomVerticalFlip(p=1.0),
            get_test_transform()
        ])
    ]
