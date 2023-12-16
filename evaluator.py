import logging
import random
import time
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from predictor import Predictor

class Evaluator:
    def __init__(self, trigger_threshold=0.5):
        self.trigger_threshold = trigger_threshold
        
    def print_results(self, methods, snapshot_sequence, test_true_labels, test_predicted_probs, start_time):
        test_predicted_labels = (test_predicted_probs > self.trigger_threshold).astype(int)
        true_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 1))
        false_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 0))
        false_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 1))
        true_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 0))
        logging.info(f'{methods} evaulation completed. Time: {time.time() - start_time:.2f}s.')
        logging.debug(f'Test: Predicted Labels: \n{test_predicted_labels}')
        logging.debug(f'Test: True Labels: \n{test_true_labels}') 
        logging.info(f'{methods}. Test: True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, True Negatives: {true_negatives}.')
        precision = precision_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
        recall = recall_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
        f1 = f1_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
        logging.warning(f'{methods}. Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}. {len(snapshot_sequence)} snapshots.')

    def get_evaulation_sequence_filenames(self, bucket, sequence_dir_path, min_nodes, max_nodes, log_window, max_sequences):
        prefix = f'{sequence_dir_path}log_window_{log_window}/'
        log_window_filtered_filenames = [blob.name for blob in bucket.list_blobs(prefix=prefix)]
        node_num_filtered_filenames = [fn for fn in log_window_filtered_filenames if int(fn.split('/')[2].split('_')[0]) >= min_nodes and int(fn.split('/')[2].split('_')[0]) <= max_nodes]
        random.shuffle(node_num_filtered_filenames)
        return node_num_filtered_filenames[:max_sequences] # Limit the number of sequences to max_sequences

    def evaluate_sequence(self, predictor, snapshot_sequence):
        start_time = time.time()
        all_predicted_labels = []
        all_true_labels = []
        i = 0
        for snapshot in snapshot_sequence:
            i += 1
            true_labels = snapshot.y
            predicted_labels = predictor.predict(snapshot)
            all_true_labels.append(true_labels.cpu().numpy())
            all_predicted_labels.append(predicted_labels.cpu().detach().numpy())
            # logging.debug(f'Snapshot {i} of {len(snapshot_sequence)} completed.')
        all_predicted_labels = np.concatenate(all_predicted_labels)
        all_true_labels = np.concatenate(all_true_labels)
        self.print_results(predictor.predictor_type, snapshot_sequence, all_true_labels, all_predicted_labels, start_time)
        return all_predicted_labels, all_true_labels

    def evaluate_test_set(self, predictor_type, predictor_filename, test_snapshot_sequence_path, bucket_name='gnn_rddl'):
        logging.info(f'Evaluating {predictor_type} predictor {predictor_filename} on {test_snapshot_sequence_path}.')
        predictor = Predictor(predictor_type, predictor_filename, bucket_name=bucket_name)
        indexed_snapshot_sequence = torch.load(test_snapshot_sequence_path)
        snapshot_sequence = indexed_snapshot_sequence[0]['snapshot_sequence']
        return self.evaluate_sequence(predictor, snapshot_sequence)

