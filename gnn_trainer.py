import time
import re
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, RGCNConv, Sequential
from torch_geometric.loader import DataLoader
from gnn import GCN, RGCN, GIN


def split_snapshots(snapshot_sequence, train_share=0.8, val_share=0.2):
    """
    Split the snapshots into training and validation sets.
    """
    n_snapshots = len(snapshot_sequence)
    n_train = int(train_share * n_snapshots)
    n_val = int(val_share * n_snapshots)

    # Ensure that we have at least one snapshot in each set
    n_train = max(1, n_train)
    n_val = max(1, n_val)

    # Shuffle and split the snapshot indices
    indices = list(range(n_snapshots))
    random.shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]

    train_snapshots = [snapshot_sequence[i] for i in train_indices]
    val_snapshots = [snapshot_sequence[i] for i in val_indices]

    return train_snapshots, val_snapshots


def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for batch in data_loader:
            out = model(batch)
            out = F.log_softmax(out, dim=1)
            loss = F.nll_loss(out, batch.y)
            total_loss += loss.item()
            predicted_labels = out.max(1)[1]
            all_predicted_labels.append(predicted_labels.cpu().numpy())
            true_labels = batch.y
            all_true_labels.append(true_labels.cpu().numpy())

    all_predicted_labels = np.concatenate(all_predicted_labels)
    all_true_labels = np.concatenate(all_true_labels)

    return total_loss / len(data_loader), all_predicted_labels, all_true_labels

def plot_training_results(filename, loss_values, val_loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('loss_curves/' + filename)
    plt.close()
    logging.info(f'Loss curves written to loss_curves/{filename}.')


def print_results(methods, snapshot_sequence, test_true_labels, test_predicted_labels, start_time):
    true_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 1))
    false_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 0))
    false_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 1))
    true_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 0))
    logging.info(f'{methods} training completed. Time: {time.time() - start_time:.2f}s.')
    logging.debug(f'Test: Predicted Labels: \n{test_predicted_labels}')
    logging.debug(f'Test: True Labels: \n{test_true_labels}') 
    logging.info(f'{methods}. Test: True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, True Negatives: {true_negatives}.')
    precision = precision_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    recall = recall_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    logging.warning(f'{methods}. Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}. {len(snapshot_sequence)} snapshots.')

def save_checkpoint(model, optimizer, epoch, loss, model_path, filename_prefix):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    filename = os.path.join(model_path, f'{filename_prefix}_checkpoint_{epoch}.pt')
    torch.save(checkpoint, filename)
    logging.info(f'Checkpoint saved: {filename}')

def train_gnn(sequence_file_name=None, 
              number_of_epochs=10, 
              max_instances=100,
              learning_rate=0.01, 
              batch_size=1, 
              hidden_layers_list=[[64, 64]],
              checkpoint_interval=1,  # Add a parameter to set checkpoint interval
              model_path='models/'):

    logging.info(f'GNN training started.')

    data = torch.load(sequence_file_name)
    if max_instances < len(data):
        data = data[:max_instances]
    snapshot_sequence = [item for sublist in [r['snapshot_sequence'] for r in data] for item in sublist]
    first_graph = snapshot_sequence[0]
    actual_num_features = first_graph.num_node_features
    num_relations = first_graph.num_edge_types
    train_snapshots, val_snapshots = split_snapshots(snapshot_sequence)

    for hidden_layers in hidden_layers_list:
        # model = GCN([actual_num_features] + hidden_layers + [2])
        model = RGCN([actual_num_features] + hidden_layers + [2], num_relations)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        train_loader = DataLoader(train_snapshots, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_snapshots, batch_size=batch_size, shuffle=False)
        n_snapshots = len(snapshot_sequence)

        loss_values, val_loss_values = [], []
        logging.info(f'Training started. Number of snapshots: {n_snapshots}. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}. Batch size: {batch_size}. Number of epochs: {number_of_epochs}.')
        for epoch in range(number_of_epochs):
            start_time = time.time()
            model.train()
            epoch_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                out = model(batch)
                out = F.log_softmax(out, dim=1)
                loss = F.nll_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            loss_values.append(epoch_loss)

            val_loss, predicted_labels, true_labels = evaluate_model(model, val_loader)
            val_loss_values.append(val_loss)
            end_time = time.time()
            logging.info(f'Epoch {epoch}: Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}. Time: {end_time - start_time:.4f}s. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}')
            if epoch % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, epoch, epoch_loss, model_path, 'latest_model_checkpoint')
                plot_training_results(f'latest_loss.png', loss_values, val_loss_values)
            
            # Early stopping
            training_stagnation_threshold = 3
            if epoch > training_stagnation_threshold:
                if val_loss > max(val_loss_values[-training_stagnation_threshold:]):
                    logging.info(f'Validation loss has not improved for {training_stagnation_threshold} epochs. Training stopped.')
                    break   



        match = re.search(r'sequence_(.*?)\.pkl', sequence_file_name)
        snapshot_name = match.group(1) if match else None
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_root = f'{snapshot_name}_hl_{hidden_layers}_n_{n_snapshots}_lr_{learning_rate}_bs_{batch_size}_{date_time_str}'
        plot_training_results(f'loss_{filename_root}.png', loss_values, val_loss_values)

        model_file_name = f'{model_path}model_{filename_root}.pt'
        torch.save(model, model_file_name)

    return model_file_name

