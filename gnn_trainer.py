import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
from simulator import produce_training_data

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

def create_masks(data_series):
    for data in data_series:
        num_nodes = data.num_nodes
        all_indices = torch.randperm(num_nodes)

        train_size = int(0.7 * num_nodes)
        val_size = int(0.15 * num_nodes)

        train_indices = all_indices[:train_size]
        val_indices = all_indices[train_size:train_size + val_size]
        test_indices = all_indices[train_size + val_size:]

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[train_indices] = True
        data.val_mask[val_indices] = True
        data.test_mask[test_indices] = True

def evaluate_model(model, data_loader, mask):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch)
            loss = F.nll_loss(out[mask], batch.y[mask])
            total_loss += loss.item()
            predicted_labels = out[mask].max(1)[1]
            all_predicted_labels.append(predicted_labels.cpu().numpy())
            true_labels = batch.y[mask]
            all_true_labels.append(true_labels.cpu().numpy())

    all_predicted_labels = np.concatenate(all_predicted_labels)
    all_true_labels = np.concatenate(all_true_labels)

    return total_loss / len(data_loader), all_predicted_labels, all_true_labels

def plot_training_results(loss_values, val_loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('loss_curve.png')
    plt.close()

def train_model(use_saved_data=True, n_simulations=10, log_window=20, max_start_time_step=30, number_of_epochs=10, debug_print=False):


    data_series = produce_training_data(use_saved_data, n_simulations, log_window, max_start_time_step, 'content/', 42, debug_print)
 
    create_masks(data_series)

    first_graph = data_series[0]
    actual_num_features = first_graph.num_node_features

    model = GCN(num_node_features=actual_num_features, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    data_loader = DataLoader(data_series, batch_size=1, shuffle=True)

    loss_values, val_loss_values = [], []
    for epoch in range(number_of_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for batch in data_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        loss_values.append(epoch_loss)

        val_loss, predicted_labels, true_labels = evaluate_model(model, data_loader, batch.val_mask)
        val_loss_values.append(val_loss)
        end_time = time.time()
        print(f'Epoch {epoch}: Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}. Time: {end_time - start_time:.4f}s')

        precision = precision_score(true_labels, predicted_labels, average='binary')
        recall = recall_score(true_labels, predicted_labels, average='binary')
        f1 = f1_score(true_labels, predicted_labels, average='binary')
        print(f'Epoch {epoch}: Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

    plot_training_results(loss_values, val_loss_values)

    test_loss, predicted_labels, true_labels = evaluate_model(model, data_loader, batch.test_mask)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test: Precision: {precision}, Recall: {recall}, F1 Score: {f1}')

train_model(use_saved_data=False, n_simulations=10, log_window=20, max_start_time_step=30, number_of_epochs=10, debug_print=False)