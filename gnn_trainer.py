import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
from simulator import produce_training_data_parallel

class GCN(torch.nn.Module):
    def __init__(self, layer_sizes):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()

        for i in range(len(layer_sizes) - 1):
            self.layers.append(GCNConv(layer_sizes[i], layer_sizes[i+1]))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  # Apply ReLU and Dropout to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        return F.log_softmax(x, dim=1)

def evaluate_model(model, data_loader, masks):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for batch, mask in zip(data_loader, masks):
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

def train_gnn(number_of_epochs=10, snapshot_sequence=None, learning_rate=0.01, hidden_layers=[16, 32, 16]):

    first_graph = snapshot_sequence[0]
    actual_num_features = first_graph.num_node_features

    model = GCN([actual_num_features] + hidden_layers + [2])
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    data_loader = DataLoader(snapshot_sequence, batch_size=1, shuffle=True)

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

        val_masks = [snapshot.val_mask for snapshot in snapshot_sequence]
        val_loss, predicted_labels, true_labels = evaluate_model(model, data_loader, val_masks)
        val_loss_values.append(val_loss)
        end_time = time.time()
        logging.info(f'Epoch {epoch}: Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}. Time: {end_time - start_time:.4f}s. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}')


    plot_training_results(loss_values, val_loss_values)

    test_masks = [snapshot.test_mask for snapshot in snapshot_sequence]
    test_loss, test_predicted_labels, test_true_labels = evaluate_model(model, data_loader, test_masks)

    return test_predicted_labels, test_true_labels

