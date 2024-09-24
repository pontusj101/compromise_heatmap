import io
import torch
from pathlib import Path
import random
import logging
import numpy as np
from google.cloud import storage
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from .bucket_manager import BucketManager

logger = logging.getLogger(__name__)

class TimeSeriesDataset(Dataset):
    def __init__(self,
                 bucket_manager,
                 file_paths,
                 max_log_window=999999,
                 online=False): # 'by_sequence', 'random'
        self.bucket_manager = bucket_manager
        self.max_log_window = max_log_window
        self.data = self.load_all_data(file_paths, online)
        self.length = len(self.data)

    def load_all_data(self, file_paths, online):
        all_snapshot_sequences = []
        for file_path in file_paths:
            snapshot_sequence = self.load_snapshot_sequence(file_path, online)
            for snapshot in snapshot_sequence:
                snapshot.x = snapshot.x[:, :self.max_log_window + 1] # Truncate the log window
            all_snapshot_sequences.append(snapshot_sequence)
        return all_snapshot_sequences

    def load_snapshot_sequence(self, file_name, online):
        if online:
            data = self.bucket_manager.torch_load_from_bucket(file_name)
        else:
            with open('data/' + file_name, 'rb') as f:
                data = torch.load(f)
        return data['snapshot_sequence']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return the idx-th time step across all series
        return self.data[idx]


def list_files_relative_to_cwd(directory):
    paths=[]
    for file in Path(directory).rglob('*'):
        if file.is_file():
            paths.append(Path(*file.parts[1:]).as_posix())  # remove data/
    return paths

def get_sequence_filenames(bucket_manager,
                           sequence_dir_path,
                           min_nodes, max_nodes,
                           min_snapshots, max_snapshots,
                           log_window,
                           max_sequences,
                           n_validation_sequences,
                           n_uncompromised_sequences,
                           fp_rate,
                           online):
    # TODO: #1 Now filtering by exact log window, but should select all above log_window, as they are truncate later anyhow.
    prefix = f'{sequence_dir_path}'
    if online:
        filenames = [blob.name for blob in bucket_manager.bucket.list_blobs(prefix=prefix)]
    else:
        filenames = list_files_relative_to_cwd('data/training_sequences')
    log_window_filtered_filenames = [fn for fn in filenames if int(fn.split('/')[1].split('_')[2]) >= log_window]
    node_num_filtered_filenames = [fn for fn in log_window_filtered_filenames if int(fn.split('/')[2].split('_')[0]) >= min_nodes and int(fn.split('/')[2].split('_')[0]) <= max_nodes]
    fp_rate_filtered_filenames = [fn for fn in node_num_filtered_filenames if float(fn.split('/')[3].split('_')[1]) == fp_rate]
    n_snapshot_filtered_filenames = [fn for fn in fp_rate_filtered_filenames if int(fn.split('/')[4].split('_')[0]) >= min_snapshots and int(fn.split('/')[4].split('_')[0]) < max_snapshots]
    random.shuffle(n_snapshot_filtered_filenames)
    uncompromised_filenames = [f for f in n_snapshot_filtered_filenames if 'passive' in f]
    compromised_filenames = [f for f in n_snapshot_filtered_filenames if 'passive' not in f]
    if len(uncompromised_filenames) < n_uncompromised_sequences:
        raise ValueError(f'Not enough uncompromised sequences. {len(uncompromised_filenames)} sequences found, but {n_uncompromised_sequences} uncompromised sequences requested.')
    filtered_filenames = uncompromised_filenames[:n_uncompromised_sequences] + compromised_filenames[:n_validation_sequences+max_sequences-n_uncompromised_sequences]
    random.shuffle(filtered_filenames)
    if len(filtered_filenames) < n_validation_sequences + 1:
        raise ValueError(f'Not enough sequences for training and validation. {len(uncompromised_filenames) + len(compromised_filenames)} sequences found, but {n_validation_sequences} validation sequences requested, and at least one additional is required for training.')
    logger.info(f'Found {len(compromised_filenames)} compromised and {len(uncompromised_filenames)} uncompromised sequences. Using {min(max_sequences, len(filtered_filenames)-n_validation_sequences)} for training and {n_validation_sequences} for validation. Using {n_uncompromised_sequences} uncompromised sequences.')
    return filtered_filenames[n_validation_sequences:n_validation_sequences+max_sequences], filtered_filenames[:n_validation_sequences] # Limit the number of sequences to max_sequences

