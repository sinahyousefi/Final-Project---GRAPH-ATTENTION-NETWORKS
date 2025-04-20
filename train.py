
import torch
import torch.nn.functional as F
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from torch_geometric.datasets import Planetoid
import time
import platform
import pandas as pd
from sklearn.manifold import TSNE
import networkx as nx

# Models
from gat import GAT
from gcn import GCN
from sage import GraphSAGE
from graphsage_mean import GraphSAGE_Mean
from graphsage_pool import GraphSAGE_Pooling
from chebnet import ChebNet
from gated_gcn import GatedGCN
from semiemb import SemiEmb

def set_seed(seed: int = 1) -> None:
    """Fix random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_eda_on_cora(dataset):
    data = dataset[0]

    print("\n=== Cora Dataset EDA ===")
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges} (undirected)")
    print(f"Number of features per node: {data.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Average degree: {data.num_edges // data.num_nodes}")

    y = data.y.cpu().numpy()
    class_counts = pd.Series(y).value_counts().sort_index()
    print("\nClass Distribution:")
    for idx, count in class_counts.items():
        print(f"  Class {idx}: {count} nodes")

    class_counts.index = [f'Class {i}' for i in class_counts.index]
    plt.figure(figsize=(8, 5))
    sns.barplot(x=class_counts.index, y=class_counts.values, hue=class_counts.index, palette="viridis", legend=False)
    plt.title("Cora Node Class Distribution")
    plt.xlabel("Class Label")
    plt.ylabel("Number of Nodes")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("plots/cora_class_distribution.png")
    print("Saved class distribution plot to: plots/cora_class_distribution.png\n")

def train(model, data, optimizer) -> float:
    """Performs one training step."""
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, data):
    """Evaluates the model on all three data splits and returns accuracy + test preds."""
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    accs = [(pred[mask] == data.y[mask]).sum().item() / mask.sum().item()
            for mask in [data.train_mask, data.val_mask, data.test_mask]]
    return accs, pred[data.test_mask].cpu(), data.y[data.test_mask].cpu()


def get_model(name: str, in_dim: int, hid_dim: int, out_dim: int, config: dict, device: torch.device):
    """Returns a model and its optimizer based on the model name."""
    if name == 'GAT':
        model = GAT(in_dim, hid_dim, out_dim, heads=config['heads'], dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam([
            {'params': model.conv1.parameters(), 'weight_decay': config['conv1_weight_decay']},
            {'params': model.conv2.parameters(), 'weight_decay': config['conv2_weight_decay']}
        ], lr=config['lr'])
    elif name == 'GCN':
        model = GCN(in_dim, hid_dim, out_dim, dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'GraphSAGE':
        model = GraphSAGE(in_dim, hid_dim, out_dim, dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'GatedGCN':
        model = GatedGCN(in_dim, hid_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'ChebNet':
        model = ChebNet(in_dim, hid_dim, out_dim, dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'SemiEmb':
        model = SemiEmb(in_dim, out_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'GraphSAGE-Mean':
        model = GraphSAGE_Mean(in_dim, hid_dim, out_dim, dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    elif name == 'GraphSAGE-Pooling':
        model = GraphSAGE_Pooling(in_dim, hid_dim, out_dim, dropout=config['dropout']).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['conv1_weight_decay'])
    else:
        raise ValueError(f"Model '{name}' is not supported.")
    return model, optimizer

def get_cora_config(model_name='GAT'):
    base = {
        'root': 'data/Cora',
        'hidden_channels': 8,
        'dropout': 0.6,
        'lr': 0.005,
        'conv1_weight_decay': 5e-4,
        'conv2_weight_decay': 0,
        'patience': 100,
        'epochs': 500
    }

    if model_name == 'GAT':
        base['heads'] = [8, 1]
        base['in_dim'] = 1433
        base['out_dim'] = 7

    elif model_name == 'GCN':
        base['hidden_channels'] = 16
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'ChebNet':
        base['hidden_channels'] = 16
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'GraphSAGE':
        base['hidden_channels'] = 128
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'GraphSAGE-Mean':
        base['hidden_channels'] = 128
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'GraphSAGE-Pooling':
        base['hidden_channels'] = 128
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'GatedGCN':
        base['hidden_channels'] = 32
        base['num_layers'] = 3
        base['dropout'] = 0.5
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    elif model_name == 'SemiEmb':
        base['lr'] = 0.01
        base['conv1_weight_decay'] = 5e-4

    return base

def plot_tsne_with_attention(model, data, model_name, dataset_name):
    model.eval()
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        out = model(x, edge_index)

        # Get embeddings before final classification (e.g., output of first GAT layer)
        if hasattr(model, 'conv1'):
            embeddings = model.conv1(x, edge_index).cpu().numpy()
        else:
            embeddings = out.cpu().numpy()

        labels = data.y.cpu().numpy()

        # Perform t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        tsne_result = tsne.fit_transform(embeddings)

        # Plot embeddings
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='tab10', s=20, alpha=0.8)
        plt.colorbar(scatter, ticks=range(data.y.max().item() + 1), label='Class')
        plt.title(f"{model_name} on {dataset_name}: t-SNE of Node Embeddings")

        # Highlight top attention edges (optional)
        if hasattr(model, 'alpha'):
            alpha_weights = model.alpha
            if isinstance(alpha_weights, torch.Tensor):
                alpha_weights = alpha_weights.detach().cpu().numpy()
            G = nx.Graph()
            tsne_coords = {i: (tsne_result[i][0], tsne_result[i][1]) for i in range(data.num_nodes)}
            G.add_nodes_from(tsne_coords)
            for i in range(edge_index.size(1)):
                src, tgt = edge_index[0, i].item(), edge_index[1, i].item()
                if src < data.num_nodes and tgt < data.num_nodes:
                    G.add_edge(src, tgt, weight=alpha_weights[i] if i < len(alpha_weights) else 0.1)
            edges = sorted(G.edges(data=True), key=lambda x: x[2]['weight'], reverse=True)[:100]
            nx.draw_networkx_edges(G, pos=tsne_coords, edgelist=[(u, v) for u, v, _ in edges], edge_color='black', alpha=0.4)

        plt.tight_layout()
        filename = f"plots/{model_name}_{dataset_name}_tsne_attention.png"
        plt.savefig(filename)
        print(f"Saved: {filename}")
        plt.close()

def plot_metrics(epoch_metrics: dict, model_name: str, dataset_name: str) -> None:
    """Plots training/validation/test accuracy and F1 scores over epochs."""
    plt.figure(figsize=(10, 6))
    for label, values in epoch_metrics.items():
        plt.plot(values, label=label)

    plt.title(f"{model_name} on {dataset_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f'plots/{model_name}_{dataset_name}_metrics.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_loss(loss_list: list[float], model_name: str, dataset_name: str) -> None:
    """Plots training loss over epochs."""
    plt.figure(figsize=(8, 5))
    plt.plot(loss_list, color='red', linewidth=2, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} on {dataset_name} — Loss vs Epoch')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    filename = f'plots/{model_name}_{dataset_name}_loss_vs_epoch.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str, dataset_name: str) -> None:
    """Plots confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    filename = f'plots/{model_name}_{dataset_name}_confusion_matrix.png'
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()


def main(dataset_name: str = 'Cora', model_name: str = 'ChebNet', seed: int = 42):
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    start_time = time.time()
    print("System Information:")
    print("Platform:", platform.platform())
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Device:", device)


    # Load dataset and config
    if dataset_name == 'Cora':
        config = get_cora_config(model_name)
    else:
        raise ValueError("Only 'Cora' is currently supported.")

    dataset = Planetoid(root=config['root'], name=dataset_name)
    data = dataset[0].to(device)
    
    data.x = data.x / (data.x.sum(1, keepdim=True) + 1e-6)

    # Setup model and optimizer
    model, optimizer = get_model(
        model_name,
        dataset.num_features,
        config['hidden_channels'],
        dataset.num_classes,
        config,
        device
    )

    # Metric tracking
    epoch_metrics = {
        'Loss': [],
        'Train Accuracy': [],
        'Val Accuracy': [],
        'Test Accuracy': [],
        'Micro-F1': [],
        'Macro-F1': []
    }

    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    if model_name == 'GAT':
        plot_tsne_with_attention(model, data, model_name, dataset_name)

    for epoch in range(config['epochs']):
        loss = train(model, data, optimizer)
        (accs, test_pred, test_true) = evaluate(model, data)
        train_acc, val_acc, test_acc = accs
        micro_f1 = f1_score(test_true, test_pred, average='micro')
        macro_f1 = f1_score(test_true, test_pred, average='macro')

        # Log metrics
        epoch_metrics['Loss'].append(loss)
        epoch_metrics['Train Accuracy'].append(train_acc)
        epoch_metrics['Val Accuracy'].append(val_acc)
        epoch_metrics['Test Accuracy'].append(test_acc)
        epoch_metrics['Micro-F1'].append(micro_f1)
        epoch_metrics['Macro-F1'].append(macro_f1)
        

        # Early Stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = model.state_dict()
            best_stats = {
                'Train Accuracy': train_acc,
                'Test Accuracy': test_acc,
                'Micro-F1': micro_f1,
                'Macro-F1': macro_f1,
                'Pred': test_pred,
                'True': test_true
            }
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Macro-F1: {macro_f1:.4f}")

        if patience_counter >= config['patience']:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model and visualize
    model.load_state_dict(best_model_state)
    selected_keys = ['Train Accuracy', 'Val Accuracy', 'Test Accuracy', 'Micro-F1', 'Macro-F1']
    filtered_metrics = {k: epoch_metrics[k] for k in selected_keys}
    run_eda_on_cora(dataset)
    plot_metrics(filtered_metrics, model_name, dataset_name)
    plot_loss(epoch_metrics['Loss'], model_name, dataset_name)
    plot_confusion_matrix(best_stats['True'], best_stats['Pred'], model_name, dataset_name)

    total_time = time.time() - start_time
    avg_epoch_time = total_time / (best_epoch + 1)

    print(f"\n--- Runtime Statistics ---")
    print(f"Total runtime (s): {total_time:.2f}")
    print(f"Average time per epoch (s): {avg_epoch_time:.4f}")
    print(f"Total training epochs (actual): {best_epoch + 1}")
    print(f"Estimated GPU hours used: {total_time / 3600:.4f} hours")

    # Final Summary
    print("\n--- Final Results ---")
    print(f"Model: {model_name} | Dataset: {dataset_name}")
    print(f"Best Epoch: {best_epoch}")
    print(f"Train Accuracy: {best_stats['Train Accuracy']:.4f}")
    print(f"Test Accuracy: {best_stats['Test Accuracy']:.4f}")
    print(f"Micro-F1: {best_stats['Micro-F1']:.4f}")
    print(f"Macro-F1: {best_stats['Macro-F1']:.4f}")

if __name__ == '__main__':
    
    dataset_name = 'Cora'
    model_name = 'GAT'  # Change to 'GAT','GCN', 'GraphSAGE', 'GatedGCN', 'ChebNet', 'SemiEmb', 'GraphSAGE-Mean', 'GraphSAGE-Pooling'
    
    
    main(dataset_name=dataset_name, model_name=model_name)

