import time
import re
import os
import io
import random
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import logging
import wandb
import hypertune
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, RGCNConv, Sequential
from .util import TimeSeriesDataset, get_sequence_filenames
from .bucket_manager import BucketManager
from torch_geometric.loader import DataLoader
from .gnn import GCN, RGCN, GIN, GAT, GNN_LSTM

logger = logging.getLogger(__name__)

def plot_training_results(bucket_manager, filename, loss_values, val_loss_values):
    plt.figure(figsize=(10, 6))
    plt.plot(loss_values, label='Training Loss')
    plt.plot(val_loss_values, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig('loss_plt.png')
    plt.close()
    bucket_manager.upload_from_filepath('loss_plt.png', filename)

def print_results(methods, snapshot_sequence, test_true_labels, test_predicted_labels, start_time):
    true_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 1))
    false_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 0))
    false_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 1))
    true_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 0))
    logger.info(f'{methods} training completed. Time: {time.time() - start_time:.2f}s.')
    logger.debug(f'Test: Predicted Labels: \n{test_predicted_labels}')
    logger.debug(f'Test: True Labels: \n{test_true_labels}')
    logger.info(f'{methods}. Test: True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, True Negatives: {true_negatives}.')
    precision = precision_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    recall = recall_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    logger.warning(f'{methods}. Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}. {len(snapshot_sequence)} snapshots.')

def save_model_to_bucket(bucket_manager, model_path, model, gnn_type, training_sequence_filenames, hidden_layers, lstm_hidden_dim, learning_rate, batch_size, snapshot_sequence_length, date_time_str):
    model_file_name = model_filename(model_path, gnn_type, training_sequence_filenames, hidden_layers, lstm_hidden_dim, learning_rate, batch_size, snapshot_sequence_length, date_time_str)
    bucket_manager.torch_save_to_bucket(model, model_file_name)
    return model_file_name

def model_filename(model_path, gnn_type, training_sequence_filenames, hidden_layers, lstm_hidden_dim, learning_rate, batch_size, snapshot_sequence_length, date_time_str):
    snapshot_name = os.path.commonprefix(training_sequence_filenames).replace('training_sequences/', '')
    filename_root = f'{gnn_type}/{snapshot_name}_hl{hidden_layers}lstm_{lstm_hidden_dim}_nsnpsht_{snapshot_sequence_length}_lr_{learning_rate:.4f}_bs_{batch_size}_{date_time_str}'
    filename_root = filename_root.replace('[', '_').replace(']', '_').replace(' ', '')
    model_file_name = f'{model_path}model/{filename_root}.pt'
    return model_file_name

def get_num_relations(bucket_manager, training_sequence_filenames):
    first_filename = training_sequence_filenames[0]
    # first_data = bucket_manager.torch_load_from_bucket(first_filename)
    # with open('data/' + first_filename, "rb") as f:
        # first_data = torch.load(f)
    # first_snapshot = first_data['snapshot_sequence'][0]
    # num_relations = first_snapshot.num_edge_types
    return 1
    # return num_relations

def make_hidden_layers(n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4):
    hidden_layers = [n_hidden_layer_1]
    if n_hidden_layer_4 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4]
    elif n_hidden_layer_3 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3]
    elif n_hidden_layer_2 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2]
    return hidden_layers


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

def attach_forward_hook(model):
    outputs = []

    def get_activation(name):
        def hook(model, input, output):
            if isinstance(output, tuple):
                outputs.append((name, output[0].detach()))
                outputs.append((name, output[1][0].detach()))
                outputs.append((name, output[1][1].detach()))
            else:
                outputs.append((name, output.detach()))
        return hook

    for name, layer in model.named_modules():
        layer.register_forward_hook(get_activation(name))

    return outputs

def get_model(gnn_type, edge_embedding_dim, heads_per_layer, actual_num_features, num_relations, hidden_layers, lstm_hidden_dim):
    if gnn_type == 'GCN':
        model = GCN([actual_num_features] + hidden_layers + [2])
    elif gnn_type == 'GIN':
        model = GIN([actual_num_features] + hidden_layers + [2])
    elif gnn_type == 'RGCN':
        model = RGCN([actual_num_features] + hidden_layers + [2], num_relations)
    elif gnn_type == 'GAT':
        heads = [heads_per_layer] * (len(hidden_layers)) + [1] # Number of attention heads in each layer
        model = GAT([actual_num_features] + hidden_layers + [2], heads, num_relations, edge_embedding_dim)
    elif gnn_type == 'GAT_LSTM':
        heads = [heads_per_layer] * (len(hidden_layers) - 1) + [1] # Number of attention heads in each layer, but setting the last layer to 1
        gnn_model = GAT([actual_num_features] + hidden_layers, heads, num_relations, edge_embedding_dim)
        model = GNN_LSTM(gnn=gnn_model, lstm_hidden_dim=lstm_hidden_dim, num_classes=2)  # Choose appropriate dimensions
    else:
        raise ValueError(f'Unknown GNN type: {gnn_type}')

    return model

def evaluate_model(model, data_loader, gnn_type, minority_weight):
    model.eval()
    total_loss = 0
    all_predicted_labels = []
    all_true_labels = []
    with torch.no_grad():
        for sequence in data_loader:
            hidden_state = None
            if gnn_type == 'GAT_LSTM':
                logits, hidden_state = model(sequence, hidden_state)
                # hidden_state = (hidden_state[0].detach(), hidden_state[1].detach()) # Detach the hidden state to prevent backpropagation through time
            else:
                logits = model(sequence)
            targets = torch.stack([snapshot.y for snapshot in sequence], dim=0).transpose(0,1)
            loss = calculate_loss(logits, targets, minority_weight)
            total_loss += loss.item()
            probabilities = F.softmax(logits, dim=-1)
            predicted_labels = torch.argmax(probabilities, dim=-1)
            all_predicted_labels.append(predicted_labels.cpu().numpy())
            all_true_labels.append(targets.cpu().numpy())

    all_predicted_labels = np.concatenate([l.flatten() for l in all_predicted_labels])
    all_true_labels = np.concatenate([l.flatten() for l in all_true_labels])

    return total_loss / len(data_loader), all_predicted_labels, all_true_labels

def calculate_loss(logits, target_labels, minority_weight):
    # Assume logits is of shape (batch_size, sequence_length, num_classes)
    # and target_labels is of shape (batch_size, sequence_length)
    # You might need to adapt this depending on how logits and target_labels are structured
    loss = 0
    for t in range(logits.shape[1]):  # Loop over each time step
        # loss += F.cross_entropy(F.log_softmax(logits[:, t, :], dim=1), target_labels[:,t], torch.Tensor([1, minority_weight]))
        loss += F.nll_loss(F.log_softmax(logits[:, t, :], dim=1), target_labels[:,t], weight=torch.Tensor([1, minority_weight]))
    return loss / logits.shape[1]  # Average loss over the sequence


def train_gnn(wandb_api_key=None,
              gnn_type='GAT',
              bucket_manager=None,
              sequence_dir_path='training_sequences/',
              model_dirpath='models/',
              number_of_epochs=8,
              max_training_sequences=99999999,
              n_validation_sequences=64,
              n_uncompromised_sequences=64,
              min_nodes=0,
              max_nodes=99999999,
                min_snapshots=0,
                max_snapshots=99999999,
              log_window=99999999,
              learning_rate=0.01,
              batch_size=1,
              n_hidden_layer_1=128,
              n_hidden_layer_2=128,
              n_hidden_layer_3=0,
              n_hidden_layer_4=0,
              edge_embedding_dim=16, # Add a parameter to set edge embedding dimension in case of GAT
              heads_per_layer=2, # Add a parameter to set number of attention heads per layer in case of GAT
              lstm_hidden_dim=128, # Add a parameter to set LSTM hidden dimension in case of GAT_LSTM
              minority_weight=10,
              checkpoint_interval=1,  # Add a parameter to set checkpoint interval
              checkpoint_file=None,  # Add checkpoint file parameter
              checkpoint_path='checkpoints/',
              online=False,
              fp_rate=0.1):

    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f'Training {gnn_type} with FP rate {fp_rate} on a maximum of {max_training_sequences} snapshot sequences for {number_of_epochs} epochs, validating on {n_validation_sequences} sequences, on graphs of sizes between {min_nodes} and {max_nodes} and sequence lengths of between {min_snapshots} and {max_snapshots} with a log window of {log_window}.')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    if gnn_type == 'GAT_LSTM':
        log_window = 1 # The LSTM will only consider the current time step, so we set the log window to 1
        batch_method = 'by_time_step' # The LSTM requires the snapshots to be ordered by time step
    # TODO: GAT should train on random, so remove this elif
    elif gnn_type == 'GAT':
        batch_method = 'by_time_step'
    else:
        batch_method = 'random'

    training_sequence_filenames, validation_sequence_filenames = get_sequence_filenames(bucket_manager, sequence_dir_path, min_nodes, max_nodes, min_snapshots, max_snapshots, log_window, max_training_sequences, n_validation_sequences, n_uncompromised_sequences, fp_rate, online)
    # TODO: #2 Check that the balance between compromised and uncompromised nodes is not too skewed. If it is, then we should sample the training data to get a more balanced dataset.
    training_data_loader = TimeSeriesDataset(bucket_manager, training_sequence_filenames, max_log_window=log_window, online=online)
    validation_data_loader = TimeSeriesDataset(bucket_manager, validation_sequence_filenames, max_log_window=log_window, online=online)

    num_relations = get_num_relations(bucket_manager, training_sequence_filenames)
    hidden_layers = make_hidden_layers(n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4)
    model = get_model(gnn_type=gnn_type,
                      edge_embedding_dim=edge_embedding_dim,
                      heads_per_layer=heads_per_layer,
                      actual_num_features=log_window + 1,
                      num_relations=num_relations,
                      hidden_layers=hidden_layers,
                      lstm_hidden_dim=lstm_hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if wandb_api_key:
        os.environ['WANDB_API_KEY'] = wandb_api_key

        wandb.init(
            project="gnn_rddl",

            config={
            "learning_rate": learning_rate,
            "gnn_type": gnn_type,
            "number_of_epochs": number_of_epochs,
            "max_training_sequences": max_training_sequences,
            "n_validation_sequences": n_validation_sequences,
            "n_uncompromised_sequences": n_uncompromised_sequences,
            "min_nodes": min_nodes,
            "max_nodes": max_nodes,
            "min_snapshots": min_snapshots,
            "max_snapshots": max_snapshots,
            "log_window": log_window,
            "batch_size": batch_size,
            "n_hidden_layer_1": n_hidden_layer_1,
            "n_hidden_layer_2": n_hidden_layer_2,
            "n_hidden_layer_3": n_hidden_layer_3,
            "n_hidden_layer_4": n_hidden_layer_4,
            "edge_embedding_dim": edge_embedding_dim,
            "heads_per_layer": heads_per_layer,
            "lstm_hidden_dim": lstm_hidden_dim
            }
        )

    train_loss_values = []
    validation_loss_values = []
    global_step = 0
    logger.info(f'Training {gnn_type} with a log window of {log_window}, {len(training_data_loader)} graphs. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}. Validating on {len(validation_data_loader)} graphs.')
    for epoch in range(number_of_epochs):
        start_time = time.time()
        model.train()
        training_loss = 0.0
        number_of_compromised_nodes = 0
        number_of_uncompromised_nodes = 0

        for i, sequence in enumerate(training_data_loader):
            hidden_state = None
            # sequence.to(device)
            for snapshot in sequence:
                number_of_compromised_nodes += torch.sum(snapshot.y == 1).item()
                number_of_uncompromised_nodes += torch.sum(snapshot.y == 0).item()
            global_step += sum([len(s.y) for s in sequence])
            optimizer.zero_grad()
            if gnn_type == 'GAT_LSTM':
                logits, hidden_state = model(sequence, hidden_state)
                # hidden_state = (hidden_state[0].detach(), hidden_state[1].detach()) # Detach the hidden state to prevent backpropagation through time
            else:
                logits = model(sequence)
            targets = torch.stack([snapshot.y for snapshot in sequence], dim=0).transpose(0,1)
            loss = calculate_loss(logits, targets, minority_weight)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            logger.debug(f'Epoch {epoch}, Batch {i}/{len(training_data_loader)}, Processed nodes: {global_step}. Training Loss: {loss.item():.4f}.')

        training_loss /= len(training_data_loader)
        train_loss_values.append(training_loss)

        validation_loss, predicted_labels, true_labels = evaluate_model(model.to(device), validation_data_loader, gnn_type, minority_weight)
        validation_loss_values.append(validation_loss)

        f1 = f1_score(true_labels, predicted_labels, average='binary', zero_division=0)
        precision = precision_score(true_labels, predicted_labels, average='binary', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='binary', zero_division=0)

        end_time = time.time()
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='F1',
            metric_value=f1,
            global_step=global_step)
        logger.info(f'Epoch {epoch}: F1: {f1:.4f}. Precision: {precision:.4f}. Recall: {recall:.4f}. Training Loss: {training_loss:.4f}. Validation Loss: {validation_loss:.4f}. {number_of_compromised_nodes} compromised nodes. {number_of_uncompromised_nodes} uncompromised nodes. Time: {end_time - start_time:.4f}s.')

        if wandb_api_key:
            wandb.log({
                "global_step": global_step,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "training_loss": training_loss,
                "validation_loss": validation_loss,
                "number_of_compromised_nodes": number_of_compromised_nodes,
                "number_of_uncompromised_nodes": number_of_uncompromised_nodes,
                "time": end_time - start_time
                })

            for name, param in model.named_parameters():
                wandb.log({f"weights/{name}": wandb.Histogram(param.data.numpy()), "global_step": global_step})
                wandb.log({f"gradients/{name}": wandb.Histogram(param.grad.numpy()), "global_step": global_step})


        mfn = model_filename(model_dirpath, gnn_type, training_sequence_filenames, hidden_layers, lstm_hidden_dim, learning_rate, batch_size, len(training_data_loader), date_time_str)
        plot_training_results(bucket_manager, f'loss_plot_{mfn}.png', train_loss_values, validation_loss_values)

    if wandb_api_key:
        wandb.finish()

    model = model.to('cpu')
    model_file_name = save_model_to_bucket(bucket_manager=bucket_manager,
                                           model_path=model_dirpath,
                                           model=model,
                                           gnn_type=gnn_type,
                                           training_sequence_filenames=training_sequence_filenames,
                                           hidden_layers=hidden_layers,
                                           lstm_hidden_dim=lstm_hidden_dim,
                                           learning_rate=learning_rate,
                                           batch_size=batch_size,
                                           snapshot_sequence_length=len(training_data_loader),
                                           date_time_str=date_time_str)

    return model_file_name

