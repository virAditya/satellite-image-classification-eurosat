# 🌍 Satellite Image Classification on EuroSAT

A PyTorch-based project for classifying satellite land use images from the **EuroSAT dataset (RGB, 10 classes)**.  
Achieves **97.23% test accuracy** without transfer learning by using custom attention and balanced architectures.

---

## 🚀 Highlights
- **Models:** 3-layer Baseline, 7-layer Attention (CBAM), 12-layer Balanced (Multi-Task)
- **Accuracy:** 97.23% overall, all classes ≥94%
- **Dataset:** EuroSAT RGB (27k images, 64×64)
- **License:** MIT

---

## ⚙️ Setup
```bash
git clone https://github.com/your-username/satellite-image-classification-eurosat.git
cd satellite-image-classification-eurosat
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
📦 Quick Start
```
python
Copy code
from utils.data_loader import load_eurosat_dataset
from models import Balanced12LayerCNN
from torch.optim import AdamW
import torch.nn as nn, torch
```
# Load data
```
python
train_ds, val_ds, test_ds = load_eurosat_dataset(download=True)
```
# Setup
```
python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Balanced12LayerCNN(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.05)
```
Train and evaluate using the provided scripts or configs in /configs.

📈 Results
|Model	| Params  |	Accuracy  |	Key Feature |
|:-------|:---------:|-----------:|:-----------------|
|Baseline |	2.1M  |	94.30% | Simple CNN |
|Attention |	7.4M  |	95.98% | CBAM |
|Balanced  | 11.2M |	97.23% | Multi-Task Attention |

Top Classes: Forest 98.6%, Industrial 98.7%, River 96.3%, Highway 96.7%

🧰 Features
Modular PyTorch models

Built-in data loader and augmentations

Metrics: Accuracy, F1, Kappa

Easy reproducibility (seed=42)

🤝 Contributing
Fork the repo

Create a branch: git checkout -b feature/new-feature

Commit: git commit -m "Add feature"

Push: git push origin feature/new-feature

Open a Pull Request

📄 License
MIT License © 2025
Built with ❤️ for computer vision enthusiasts.

yaml
Copy code
