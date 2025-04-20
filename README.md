
# 🔍 Final-Project---Graph Neural Network Benchmark on Cora Dataset

This repository contains a benchmarking pipeline for evaluating various Graph Neural Network (GNN) architectures on the **Cora** dataset using PyTorch Geometric. Models include GCN, GAT, GraphSAGE (with different aggregation strategies), ChebNet, GatedGCN, and SemiEmb.

## 📚 Overview

The script supports:
- Training and evaluation of multiple GNN models
- Early stopping based on validation accuracy
- Performance metrics: Accuracy, Micro/Macro F1-Score
- Visualization: Loss curves, Accuracy/F1 over epochs, Confusion matrix

## 🧠 Implemented Models

- **GCN** — Graph Convolutional Network
- **GAT** — Graph Attention Network
- **GraphSAGE** — Neighborhood aggregation (mean, pooling)
- **ChebNet** — Chebyshev Spectral CNN
- **GatedGCN** — Gated Graph Convolution
- **SemiEmb** — Semi-supervised Embedding-based GNN

## 📦 Dependencies

Make sure to install the following Python packages:

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install matplotlib seaborn scikit-learn
```

> ✅ This code requires PyTorch Geometric and a CUDA-compatible GPU for optimal performance.

## 📁 File Structure

```
├── gat.py                   # GAT model
├── gcn.py                   # GCN model
├── sage.py                  # GraphSAGE model (base)
├── graphsage_mean.py        # GraphSAGE with mean aggregation
├── graphsage_pool.py        # GraphSAGE with pooling
├── chebnet.py               # ChebNet model
├── gated_gcn.py             # GatedGCN model
├── semiemb.py               # SemiEmb model
├── train.py                 # Main training/evaluation loop
├── plots/                   # Output folder for plots
└── README.md                # This file
```

## 🚀 How to Run

To train and evaluate a model, run:

```bash
python train.py
```

By default, it runs `GatedGCN` on the **Cora** dataset. You can change the model name in the script:

```python
model_name = 'GAT'  # Options: 'GAT', 'GCN', 'GraphSAGE', 'GatedGCN', 'ChebNet', 'SemiEmb', 'GraphSAGE-Mean', 'GraphSAGE-Pooling'
```

## 📊 Output

After training, the script generates:

- Accuracy and F1-score plots (`plots/{model}_{dataset}_metrics.png`)
- Training loss plot (`plots/{model}_{dataset}_loss_vs_epoch.png`)
- Confusion matrix heatmap (`plots/{model}_{dataset}_confusion_matrix.png`)

Example output:

```
--- Final Results ---
Model: GAT | Dataset: Cora
Best Epoch: 143
Train Accuracy: 0.9801
Test Accuracy: 0.8253
Micro-F1: 0.8237
Macro-F1: 0.8224
```

## 🧪 Dataset

This project uses the [Cora](https://linqs.soe.ucsc.edu/data) citation dataset through `torch_geometric.datasets.Planetoid`.

## ✨ Features

- Modular model selection
- Configurable training parameters
- Visual performance tracking
- Reproducible results via random seed setting

## 🛠 To Do

- Add support for other datasets (Citeseer, Pubmed, etc.)
- Include hyperparameter tuning
- Integrate with Weights & Biases or TensorBoard for logging
