import logging
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from tabular_trainer import train_tabular

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

def evaluate(snapshot_sequence=None):
    logging.info(f'Tabular training started.')
    start_time = time.time()
    test_true_labels, test_predicted_labels = train_tabular(snapshot_sequence=snapshot_sequence)
    print_results('Tabular', snapshot_sequence, test_true_labels, test_predicted_labels, start_time)
