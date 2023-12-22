import io
import torch
import random
from google.cloud import storage
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from bucket_manager import BucketManager


class TimeSeriesDataset(Dataset):
    def __init__(self, 
                 bucket_manager, 
                 file_paths, 
                 max_log_window=999999,
                 batch_size=1,
                 batch_method='by_time_step'): # 'by_sequence', 'random'
        self.bucket_manager = bucket_manager
        self.max_log_window = max_log_window
        self.batch_size = batch_size
        self.batch_method = batch_method
        self.data = self.load_all_data(file_paths)
        self.length = len(self.data[0]) 

    def load_all_data(self, file_paths):
        all_snapshot_sequences = []
        for file_path in file_paths:
            snapshot_sequence = self.load_snapshot_sequence(file_path)
            for snapshot in snapshot_sequence:
                snapshot.x = snapshot.x[:, :self.max_log_window + 1] # Truncate the log window
            all_snapshot_sequences.append(snapshot_sequence)
        self.common_snapshot_sequence_length = min(len(seq) for seq in all_snapshot_sequences) # Truncate all sequences to the length of the shortest one
        adjusted_sequences = [seq[:self.common_snapshot_sequence_length] for seq in all_snapshot_sequences]
        if self.batch_method == 'by_time_step':
            batch_list = list(zip(*adjusted_sequences))  # Transpose to group by time step
            return [Batch.from_data_list(batch) for batch in batch_list]
        elif self.batch_method == 'by_sequence':
            batch_list = adjusted_sequences
            return [Batch.from_data_list(batch) for batch in batch_list]
        elif self.batch_method == 'random':
            flattened_adjusted_sequences = [item for sublist in adjusted_sequences for item in sublist]
            random.shuffle(flattened_adjusted_sequences)
            random_batch_list = [Batch.from_data_list(flattened_adjusted_sequences[i:i + self.batch_size]) for i in range(0, len(flattened_adjusted_sequences), self.batch_size)]
            return random_batch_list
        else:
            raise ValueError(f'Batch method {self.batch_method} not supported.')

    def load_snapshot_sequence(self, file_name):
        data = self.bucket_manager.torch_load_from_bucket(file_name)
        return data['snapshot_sequence']

    def snapshot_sequence_length(self):
        return self.common_snapshot_sequence_length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return the idx-th time step across all series
        return self.data[idx]


def get_sequence_filenames(bucket_manager, sequence_dir_path, min_nodes, max_nodes, min_snapshots, max_snapshots, log_window, max_sequences, n_validation_sequences):
    # TODO: #1 Now filtering by exact log window, but should select all above log_window, as they are truncate later anyhow.
    prefix = f'{sequence_dir_path}'
    filenames = [blob.name for blob in bucket_manager.bucket.list_blobs(prefix=prefix)]
    log_window_filtered_filenames = [fn for fn in filenames if int(fn.split('/')[1].split('_')[2]) >= log_window]
    node_num_filtered_filenames = [fn for fn in log_window_filtered_filenames if int(fn.split('/')[2].split('_')[0]) >= min_nodes and int(fn.split('/')[2].split('_')[0]) <= max_nodes]
    n_snapshot_filtered_filenames = [fn for fn in node_num_filtered_filenames if int(fn.split('/')[3].split('_')[0]) >= min_snapshots and int(fn.split('/')[3].split('_')[0]) < max_snapshots]
    random.shuffle(n_snapshot_filtered_filenames)
    if len(n_snapshot_filtered_filenames) < n_validation_sequences + 1:
        raise ValueError(f'Not enough sequences for training and validation. {len(n_snapshot_filtered_filenames)} sequences found, but {n_validation_sequences} validation sequences requested, and at least one additional is required for training.')
    return n_snapshot_filtered_filenames[n_validation_sequences:n_validation_sequences+max_sequences:], n_snapshot_filtered_filenames[:n_validation_sequences] # Limit the number of sequences to max_sequences

