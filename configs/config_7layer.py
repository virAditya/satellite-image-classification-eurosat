# 7-Layer Attention CNN Config
model:
  name: 'attention_7layer'
  num_classes: 10
  dropout_rate: 0.4

training:
  num_epochs: 40
  learning_rate: 0.001
  weight_decay: 1e-4
  scheduler: 'CosineAnnealingLR'
  t_max: 40
  batch_size: 64
  early_stopping_patience: 12

data:
  batch_size: 64
  num_workers: 4
  random_seed: 42
  download: true

device: 'cuda' if torch.cuda.is_available() else 'cpu'
