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
import hypertune
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, RGCNConv, Sequential
from torch_geometric.loader import DataLoader
from gnn import GCN, RGCN, GIN, GAT


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

def train_gnn(gnn_type='GAT',
              bucket_name='gnn_rddl',
              sequence_dir_path='training_sequences/',
              sequence_file_name=None, 
              number_of_epochs=8, 
              max_sequences=99999999,
              min_nodes=0,
              max_nodes=99999999,
              log_window=99999999,
              learning_rate=0.01, 
              batch_size=1, 
              n_hidden_layer_1=128,
              n_hidden_layer_2=128,
              n_hidden_layer_3=0,
              n_hidden_layer_4=0,
              edge_embedding_dim=16, # Add a parameter to set edge embedding dimension in case of GAT
              heads_per_layer=2, # Add a parameter to set number of attention heads per layer in case of GAT
              checkpoint_interval=1,  # Add a parameter to set checkpoint interval
              checkpoint_file=None,  # Add checkpoint file parameter
              model_path='models/',
              checkpoint_path='checkpoints/'):

    logging.info(f'GNN training started.')

    logging.info(f'Loading the snapshot sequence file...')
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    prefix = f'{sequence_dir_path}log_window_{log_window}/'
    log_window_filtered_filenames = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
    node_num_filtered_filenames = [fn for fn in log_window_filtered_filenames if int(fn.split('/')[2].split('_')[0]) >= min_nodes and int(fn.split('/')[2].split('_')[0]) <= max_nodes]
    random.shuffle(node_num_filtered_filenames)
    filenames = node_num_filtered_filenames[:max_sequences]

    first_filename = filenames[0]
    blob = bucket.blob(first_filename)
    blob.download_to_filename(f'/tmp/{first_filename}')
    first_data = torch.load(f'/tmp/{first_filename}')
    first_snapshot = first_data[0]['snapshot_sequence'][0]
    actual_num_features = first_snapshot.num_node_features


    blob = bucket.blob(sequence_file_name)
    blob.download_to_filename('/tmp/snapshot_sequence.pkl')
    data = torch.load('/tmp/snapshot_sequence.pkl')
    logging.info(f'Loaded.')

    if max_sequences < len(data):
        data = data[:max_sequences]
    snapshot_sequence = [item for sublist in [r['snapshot_sequence'] for r in data] for item in sublist]
    for snapshot in snapshot_sequence:
        snapshot.x = snapshot.x[:,:log_window]
    first_graph = snapshot_sequence[0]
    actual_num_features = first_graph.num_node_features + 1
    num_relations = first_graph.num_edge_types
    total_number_of_nodes = sum([len(snapshot.y) for snapshot in snapshot_sequence])
    total_number_of_compromised_nodes = int(sum([sum(snapshot.y) for snapshot in snapshot_sequence]))

    train_snapshots, val_snapshots = split_snapshots(snapshot_sequence)

    hidden_layers = [n_hidden_layer_1]
    if n_hidden_layer_4 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3, n_hidden_layer_4]
    elif n_hidden_layer_3 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2, n_hidden_layer_3]
    elif n_hidden_layer_2 > 0:
        hidden_layers = [n_hidden_layer_1, n_hidden_layer_2]

    if gnn_type == 'GCN':
        model = GCN([actual_num_features] + hidden_layers + [2])
    elif gnn_type == 'GIN':
        model = GIN([actual_num_features] + hidden_layers + [2])
    elif gnn_type == 'RGCN':
        model = RGCN([actual_num_features] + hidden_layers + [2], num_relations)
    elif gnn_type == 'GAT':
        heads = [heads_per_layer] * (len(hidden_layers)) + [1] # Number of attention heads in each layer
        model = GAT([actual_num_features] + hidden_layers + [2], heads, num_relations, edge_embedding_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    # if checkpoint_file and os.path.isfile(checkpoint_file):
    #     logging.info(f'Attempting to load model from {checkpoint_file}.')
    #     blob = bucket.blob(checkpoint_file)
    #     blob.download_to_filename('/tmp/checkpoint.pkl')
    #     checkpoint = torch.load('/tmp/checkpoint.pkl')
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     logging.info(f'Resuming training from epoch {start_epoch}')

    train_loader = DataLoader(train_snapshots, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_snapshots, batch_size=batch_size, shuffle=False)

    loss_values, val_loss_values = [], []
    global_step = 0
    logging.info(f'Training started. Total number of nodes: {total_number_of_nodes} of which {total_number_of_compromised_nodes} were compromised. Number of snapshots: {len(snapshot_sequence)}. Log window: {first_graph.x.shape[1]} Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}. Batch size: {batch_size}. Number of epochs: {number_of_epochs}, Edge embedding dimension: {edge_embedding_dim}.')
    for epoch in range(start_epoch, number_of_epochs):
        start_time = time.time()
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            global_step += len(batch.y)
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
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag='validation_loss',
            metric_value=val_loss,
            global_step=global_step)
        logging.info(f'Epoch {epoch}: Training Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}. Time: {end_time - start_time:.4f}s. Learning rate: {learning_rate}. Hidden Layers: {hidden_layers}')
        # if epoch % checkpoint_interval == 0:
        #     checkpoint = {
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': loss,
        #     }
        #     match = re.search(r'sequence_(.*?)\.pkl', sequence_file_name)
        #     snapshot_name = match.group(1) if match else None
        #     date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     filename = f'{checkpoint_path}checkpoint_{snapshot_name}_hl_{hidden_layers}_nsnpsht_{len(snapshot_sequence)}_lr_{learning_rate}_bs_{batch_size}_{date_time_str}'

        #     buffer = io.BytesIO()
        #     torch.save(checkpoint, buffer)
        #     buffer.seek(0)
        #     blob = bucket.blob(filename)
        #     modified_retry = DEFAULT_RETRY.with_deadline(60)
        #     modified_retry = modified_retry.with_delay(initial=0.5, multiplier=1.2, maximum=10.0)
        #     blob.upload_from_file(buffer, retry=modified_retry)
        #     buffer.close()

            # plot_training_results(f'latest_loss.png', loss_values, val_loss_values)
        
        # Early stopping
        training_stagnation_threshold = 3
        if epoch > training_stagnation_threshold:
            if val_loss > max(val_loss_values[-training_stagnation_threshold:]):
                logging.info(f'Validation loss has not improved for {training_stagnation_threshold} epochs. Training stopped.')
                break   

    match = re.search(r'sequence_(.*?)\.pkl', sequence_file_name)
    snapshot_name = match.group(1) if match else None
    date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_root = f'{snapshot_name}_hl_{hidden_layers}_nsnpsht_{len(snapshot_sequence)}_lr_{learning_rate}_bs_{batch_size}_{date_time_str}'
    # plot_training_results(f'loss_{filename_root}.png', loss_values, val_loss_values)

    torch.save(model, 'local_model.pt')
    model_file_name = f'{model_path}model_{filename_root}.pt'
    blob = bucket.blob(model_file_name)
    blob.upload_from_filename('local_model.pt')

    hr = {'model_file_name': model_file_name, 'max_instances': max_sequences, 'hidden_layers': hidden_layers, 'epoch_loss': epoch_loss, 'val_loss': val_loss}
    logging.info(hr)

    return model_file_name

