import io
import torch
import random
from google.cloud import storage


def get_sequence_filenames(bucket_manager, sequence_dir_path, min_nodes, max_nodes, log_window, max_sequences):
    prefix = f'{sequence_dir_path}log_window_{log_window}/'
    log_window_filtered_filenames = [blob.name for blob in bucket_manager.bucket.list_blobs(prefix=prefix)]
    node_num_filtered_filenames = [fn for fn in log_window_filtered_filenames if int(fn.split('/')[2].split('_')[0]) >= min_nodes and int(fn.split('/')[2].split('_')[0]) <= max_nodes]
    random.shuffle(node_num_filtered_filenames)
    return node_num_filtered_filenames[:max_sequences] # Limit the number of sequences to max_sequences

