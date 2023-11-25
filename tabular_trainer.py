import logging
import torch
import numpy as np
from graph_index import GraphIndex
from torch_geometric.loader import DataLoader
from simulator import produce_training_data_parallel

def frequency(target_log_sequence, snapshot_sequence, train_masks):
    n_labels = len(snapshot_sequence[0].y)
    count = torch.zeros(n_labels)
    hits = torch.zeros(n_labels)
    for snapshot, mask in zip(snapshot_sequence, train_masks):
        log_sequence = snapshot.x[:, 1:]
        if torch.equal(log_sequence, target_log_sequence):
            for label_index in range(n_labels):
                if mask[label_index]:
                    count[label_index] += 1
                    labels = snapshot.y
                    if labels[label_index] == 1:
                        hits[label_index] += 1
    
    return torch.round(torch.nan_to_num(hits/count))

def evaluate_model(data_loader, snapshot_sequence):
    train_masks = [snapshot.test_mask for snapshot in snapshot_sequence]
    test_masks = [snapshot.test_mask for snapshot in snapshot_sequence]
    all_predicted_labels = []
    all_true_labels = []
    i = 0
    for batch, mask in zip(data_loader, test_masks):
        i += 1
        true_labels = batch.y[mask]
        unmasked_predicted_labels = frequency(batch.x[:,1:], snapshot_sequence, train_masks)
        predicted_labels = unmasked_predicted_labels[mask]
        all_true_labels.append(true_labels.cpu().numpy())
        all_predicted_labels.append(predicted_labels.cpu().numpy())
        logging.debug(f'Batch {i} of {len(test_masks)} completed.')


    all_predicted_labels = np.concatenate(all_predicted_labels)
    all_true_labels = np.concatenate(all_true_labels)

    return all_predicted_labels, all_true_labels


def train_tabular(snapshot_sequence=None, graph_size='small'):

    logging.info(f'Training tabular model on {graph_size} graph and {len(snapshot_sequence)} snapshots.')

    data_loader = DataLoader(snapshot_sequence, batch_size=1, shuffle=True)
    test_predicted_labels, test_true_labels = evaluate_model(data_loader, snapshot_sequence)

    logging.info(f'Tabular training completed.')
    return test_true_labels, test_predicted_labels
