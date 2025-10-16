**Satellite Image Classification on EuroSAT 🛰️**
Python 3.8+
PyTorch
License: MIT
Test Accuracy

A systematic evolution of custom CNN architectures for satellite land use classification on the EuroSAT dataset. Achieves 97.23% test accuracy without pre-trained models or transfer learning—through targeted attention mechanisms and balanced design.

This repo provides modular PyTorch models and utilities to go from a simple baseline (94.30%) to a production-ready model (97.23%) by addressing specific failure modes like River/Highway confusion.

***🎯 Project Highlights***
Three Models: Baseline → Attention-Enhanced → Balanced Multi-Task.

From Scratch: No ImageNet pre-training; pure architectural innovation.

Reproducible: Fixed seeds, YAML configs, full evaluation pipeline.

Results: 97.23% accuracy, all classes ≥94%, Cohen's Kappa 0.9692.

Dataset: EuroSAT RGB (27,000 Sentinel-2 images, 64×64, 10 classes).

***Performance Summary***
Model	Layers	Parameters	Test Accuracy	Key Innovation
Baseline	3	2.1M	94.30%	Simple CNN
Attention	7	7.4M	95.98%	CBAM Attention
Balanced	12	11.2M	97.23%	Multi-Task Attention
Per-Class (12-Layer Model): All ≥94.46% (Forest: 98.64%, Industrial: 98.68%, PermanentCrop: 94.46%).

***🚀 Quick Start***
1. Clone and Setup
bash
git clone https://github.com/your-username/satellite-image-classification-eurosat.git
cd satellite-image-classification-eurosat
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
2. Download Dataset
The EuroSAT dataset will be downloaded automatically when using the data loader:

python
from utils.data_loader import load_eurosat_dataset
train_ds, val_ds, test_ds = load_eurosat_dataset(download=True)  # Downloads ~90MB
print(f"Loaded: Train {len(train_ds)}, Val {len(val_ds)}, Test {len(test_ds)}")
3. Use Models and Utils
Import and use the modules directly in your Python script or Jupyter notebook:

python
import torch
import torch.nn as nn
from torch.optim import AdamW
from models import Balanced12LayerCNN
from utils import load_eurosat_dataset, get_train_transform, get_test_transform, create_loaders
from utils.metrics import compute_metrics
from utils.visualization import plot_confusion_matrix
from utils.data_loader import EuroSATConfig

# Load data with transforms
train_ds, val_ds, test_ds = load_eurosat_dataset(download=True)
train_loader, val_loader, test_loader = create_loaders(
    train_ds, val_ds, test_ds, get_train_transform(), get_test_transform()
)

# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Balanced12LayerCNN(num_classes=EuroSATConfig.NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)

# Training loop (simplified - see configs for full params)
model.train()
for epoch in range(1, 81):
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
all_preds, all_targets = [], []
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_targets.extend(targets.numpy())

# Compute and plot metrics
metrics = compute_metrics(all_targets, all_preds)
print(f"Test Accuracy: {metrics['accuracy']:.2f}%")
plot_confusion_matrix(all_targets, all_preds)

# Save model
torch.save(model.state_dict(), 'best_model.pth')
print(f"Model saved. Params: {model.get_num_params():,}")
Tips:

Use YAML configs (e.g., configs/config_12layer.yaml) to load hyperparameters programmatically.

For other models: Replace Balanced12LayerCNN with Baseline3LayerCNN or Attention7LayerCNN.

Augmentation: Use get_train_transform() for robust training.

Metrics: compute_metrics() returns accuracy, F1, Kappa, per-class details.

4. Test a Model (Standalone)
bash
python models/balanced_12layer.py  # Outputs shape, params, attention weights
Expected Output: ~11.2M params, attention α ≈0.5 (learnable).

***📊 Results***
Key Learnings:

Single attention (CBAM) fixes specific issues but creates trade-offs.

Balanced multi-task attention (Coordinate + SE) achieves robust performance.

Progressive DropBlock > standard dropout for CNNs.

Detailed Metrics (12-Layer Model):

Overall Accuracy: 97.23%

Cohen's Kappa: 0.9692

Matthews Correlation: 0.9692

Confidence Gap: 24.25% (model "knows when it knows")

Per-Class Accuracies:

Forest: 98.64%

Industrial: 98.68%

SeaLake: 98.28%

HerbaceousVegetation: 98.25%

Residential: 97.78%

Pasture: 97.66%

Highway: 96.68%

River: 96.27%

AnnualCrop: 95.55%

PermanentCrop: 94.46% (weakest, vegetation confusion)

Top Misclassifications Reduced:

River ↔ Highway: -81% (from 27 to 5 errors).

All classes balanced ≥94%.

Reproduce by training with the example code above (expected: 97.23% on test set, seed=42).

***🛠️ Features***
Models: 3-layer baseline, 7-layer CBAM, 12-layer balanced (residual + multi-task attention).

Utils: Custom data loading, augmentation (rotation, flips, jitter), metrics (F1, Kappa), visualization (Seaborn/Heatmaps).

Training: AdamW optimizer, cosine scheduling, early stopping, mixed precision (FP16) support.

Evaluation: Per-class metrics, confusion analysis, TTA (Test-Time Augmentation) via utils.

Reproducibility: Seed=42, 70/15/15 split, deterministic ops.

***📚 Technical Specs***
Dataset: EuroSAT RGB (Sentinel-2, 10 land cover classes).

Input: 64×64×3 RGB images.

Hardware: Single NVIDIA GPU (T4/P100, 6-8GB VRAM).

Time: 1.5h (3-layer), 3.5h (12-layer).

Augmentation: Rotations, flips, color jitter, Gaussian blur.

🤝 Contributing
Fork the repo.

Create a feature branch (git checkout -b feature/amazing-feature).

Commit changes (git commit -m 'Add some amazing feature').

Push to branch (git push origin feature/amazing-feature).

Open a Pull Request.

📄 License
MIT License - see LICENSE for details.

🙏 Acknowledgments
EuroSAT Dataset: Helber et al., 2019 (paper).

CBAM: Woo et al., ECCV 2018.

Coordinate Attention: Hou et al., TPAMI 2022.

PyTorch Team: For the excellent framework.


⭐ Star if helpful! Questions? Open an issue.

Built with ❤️ for computer vision enthusiasts.
