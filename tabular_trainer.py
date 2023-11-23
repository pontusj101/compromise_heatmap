import logging
import torch
import numpy as np
from graph_index import GraphIndex
from sklearn.metrics import precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
from simulator import produce_training_data_parallel

# def get_unique_snapshots(snpshot_sequence):
#     seen = set()
#     unique_log_sequences = []
#     for snapshot in snpshot_sequence:
#         log_sequence = snapshot.x[:, 1:]
#         # Convert tensor to a string for hashing. 
#         # Since we only have 0s and 1s, this is reliable.

#         log_sequence_str = str(log_sequence.tolist())

#         if log_sequence_str not in seen:
#             seen.add(log_sequence_str)
#             unique_log_sequences.append(snapshot)

#     return unique_log_sequences

def frequency(target_log_sequence, snapshot_sequence):
    n_labels = len(snapshot_sequence[0].y)
    count = 0
    hits = torch.zeros(n_labels)
    for snapshot in snapshot_sequence:
        log_sequence = snapshot.x[:, 1:]
        if torch.equal(log_sequence, target_log_sequence):
            count += 1
            for label_index in range(n_labels):
                labels = snapshot.y
                if labels[label_index] == 1:
                    hits[label_index] += 1
    return np.round(hits/count)

def evaluate_model(data_loader, masks, snapshot_sequence):
    all_predicted_labels = []
    all_true_labels = []
    for batch, mask in zip(data_loader, masks):
        true_labels = batch.y[mask]
        unmasked_predicted_labels = frequency(batch.x[:,1:], snapshot_sequence)
        logging.info(f'Unmasked predicted labels: {unmasked_predicted_labels}')
        predicted_labels = unmasked_predicted_labels[mask]
        all_true_labels.append(true_labels.cpu().numpy())
        all_predicted_labels.append(predicted_labels.cpu().numpy())

    all_predicted_labels = np.concatenate(all_predicted_labels)
    all_true_labels = np.concatenate(all_true_labels)

    return all_predicted_labels, all_true_labels


def train_tabular(snapshot_sequence=None, graph_size='small'):
    # graph_index = GraphIndex(size=graph_size)
    # n_attacksteps = len(graph_index.attackstep_mapping)
    # labels = torch.zeros(n_attacksteps)
    # logging.info(f'Number of snapshots : {len(snapshot_sequence)}')
    # unique_snapshots = get_unique_snapshots(snapshot_sequence)
    # logging.info(f'Number of unique snapshots: {len(unique_snapshots)}')
    # for snapshot in unique_snapshots:
    #     log_sequence = snapshot.x[:, 1:]
    #     logging.info(f'\n{log_sequence}')
    #     for label_index in range(n_attacksteps):
    #         labels[label_index] = frequency(log_sequence, snapshot_sequence, label_index)
    #         logging.info(f'Label {label_index}: {labels[label_index]:.2f}')

    data_loader = DataLoader(snapshot_sequence, batch_size=1, shuffle=True)

    test_masks = [snapshot.test_mask for snapshot in snapshot_sequence]
    test_predicted_labels, test_true_labels = evaluate_model(data_loader, test_masks, snapshot_sequence)
    logging.info(f'Test: Predicted Labels: \n{test_predicted_labels}')
    logging.info(f'Test: True Labels: \n{test_true_labels}') 
    precision = precision_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    recall = recall_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    logging.warning(f'Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}.')
