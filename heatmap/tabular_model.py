import logging
import torch
import numpy as np
from .graph_index import GraphIndex
from torch_geometric.loader import DataLoader

class TabularModel:
    def __init__(self, snapshot_sequence):
        self.snapshot_sequence = snapshot_sequence

    def predict(self, snapshot):
        return torch.round(self.frequency(snapshot.x[:, 1:]))

    def frequency(self, target_log_sequence, snapshot_sequence, train_masks):
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
        return torch.nan_to_num(hits/count)

    def evaluate_model(self, snapshot_sequence):
        all_predicted_labels = []
        all_true_labels = []
        i = 0
        for snapshot in snapshot_sequence:
            i += 1
            true_labels = snapshot.y[snapshot.test_mask]
            unmasked_predicted_labels = self.predict(snapshot_sequence)
            predicted_labels = unmasked_predicted_labels[snapshot.test_mask]
            all_true_labels.append(true_labels.cpu().numpy())
            all_predicted_labels.append(predicted_labels.cpu().numpy())
            logging.debug(f'Snapshot {i} of {len(snapshot_sequence)} completed.')
        all_predicted_labels = np.concatenate(all_predicted_labels)
        all_true_labels = np.concatenate(all_true_labels)
        return all_predicted_labels, all_true_labels


