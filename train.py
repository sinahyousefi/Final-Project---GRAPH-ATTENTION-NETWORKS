import torch
import torch.nn.functional as F
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from torch_geometric.datasets import Planetoid

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
        base['in_dim']=1433
        base['out_dim']=7
    elif model_name == 'GCN':
        base['hidden_channels'] = 16
    elif model_name == 'LSTM':
        base['dropout'] = 0.4
        base['lr'] = 0.002
    return base

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
        

        # Early stopping
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
    plot_metrics(filtered_metrics, model_name, dataset_name)
    plot_loss(epoch_metrics['Loss'], model_name, dataset_name)
    plot_confusion_matrix(best_stats['True'], best_stats['Pred'], model_name, dataset_name)

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
    model_name = 'GatedGCN'  # Change to 'GAT','GCN', 'GraphSAGE', 'GatedGCN', 'ChebNet', 'SemiEmb', 'GraphSAGE-Mean', 'GraphSAGE-Pooling'
    main(dataset_name=dataset_name, model_name=model_name)
