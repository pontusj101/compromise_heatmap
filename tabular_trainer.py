import logging
import torch

from simulator import produce_training_data_parallel

def frequency(target_log_sequence, snapshot_sequence):
    count = 0
    for snapshot in snapshot_sequence:
        log_sequence = snapshot.x[:, 1:]
        if torch.equal(log_sequence, target_log_sequence):
            count += 1
    return count/len(snapshot_sequence)

def get_unique_snapshots(snpshot_sequence):
    seen = set()
    unique_log_sequences = []
    for snapshot in snpshot_sequence:
        log_sequence = snapshot.x[:, 1:]
        # Convert tensor to a string for hashing. 
        # Since we only have 0s and 1s, this is reliable.

        log_sequence_str = str(log_sequence.tolist())

        if log_sequence_str not in seen:
            seen.add(log_sequence_str)
            unique_log_sequences.append(snapshot)

    return unique_log_sequences

def train_tabular(snapshot_sequence=None):
    logging.info(f'Number of snapshots : {len(snapshot_sequence)}')
    unique_snapshots = get_unique_snapshots(snapshot_sequence)
    logging.info(f'Number of unique snapshots: {len(unique_snapshots)}')
    for snapshot in unique_snapshots:
        log_sequence = snapshot.x[:, 1:]
        f = frequency(log_sequence, snapshot_sequence)
        logging.info(f'Frequency of log sequence \n{log_sequence}: {f:.2f}')