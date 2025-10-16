"""
EuroSAT CNN Models Package
Three progressive architectures for satellite image classification.
"""

from .baseline_3layer import Baseline3LayerCNN
from .attention_7layer import Attention7LayerCNN
from .balanced_12layer import Balanced12LayerCNN

__all__ = ['Baseline3LayerCNN', 'Attention7LayerCNN', 'Balanced12LayerCNN']
__version__ = '1.0.0'
