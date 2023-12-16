import logging
import random
import time
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from predictor import Predictor
from util import get_sequence_filenames

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

    def evaluate_test_set(self, 
                          predictor_type, 
                          predictor_filename, 
                          bucket_manager, 
                          sequence_dir_path, 
                          min_nodes, 
                          max_nodes, 
                          log_window, 
                          max_sequences):
        predictor = Predictor(predictor_type, predictor_filename, bucket_manager=bucket_manager)
        sequence_filenames = get_sequence_filenames(bucket_manager, sequence_dir_path, min_nodes, max_nodes, log_window, max_sequences)
        logging.info(f'Evaluating {predictor_type} predictor {predictor_filename} on {len(sequence_filenames)} snapshot sequences.')
        start_time = time.time()
        all_predicted_labels = []
        all_true_labels = []
        i = 0
        for sequence_filename in sequence_filenames:
            data = bucket_manager.torch_load_from_bucket(sequence_filename)
            snapshot_sequence = data['snapshot_sequence']
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

