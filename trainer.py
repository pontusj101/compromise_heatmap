import logging
import torch
import numpy as np
# import cProfile
# import pstats
# import io
from sklearn.metrics import precision_score, recall_score, f1_score
from simulator import produce_training_data_parallel
from tabular_trainer import train_tabular
from gnn_trainer import train_gnn

def create_masks(snapshot_sequence, train_share=0.7, val_share=0.15, test_share=0.15):
    assert train_share + val_share + test_share == 1
    for snapshot in snapshot_sequence:
        num_nodes = snapshot.num_nodes
        all_indices = torch.randperm(num_nodes)

        test_size = int(np.ceil(test_share * num_nodes))
        val_size = int(np.ceil(val_share * num_nodes))

        test_indices = all_indices[:test_size]
        val_indices = all_indices[test_size:test_size + val_size]
        train_indices = all_indices[test_size + val_size:]

        snapshot.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        snapshot.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        snapshot.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        snapshot.train_mask[train_indices] = True
        snapshot.val_mask[val_indices] = True
        snapshot.test_mask[test_indices] = True

def print_results(methods, snapshot_sequence, test_true_labels, test_predicted_labels):
    true_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 1))
    false_positives = np.sum(np.logical_and(test_predicted_labels == 1, test_true_labels == 0))
    false_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 1))
    true_negatives = np.sum(np.logical_and(test_predicted_labels == 0, test_true_labels == 0))
    logging.debug(f'Test: Predicted Labels: \n{test_predicted_labels}')
    logging.debug(f'Test: True Labels: \n{test_true_labels}') 
    logging.info(f'{methods}. Test: True Positives: {true_positives}, False Positives: {false_positives}, False Negatives: {false_negatives}, True Negatives: {true_negatives}.')
    precision = precision_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    recall = recall_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    f1 = f1_score(test_true_labels, test_predicted_labels, average='binary', zero_division=0)
    logging.warning(f'{methods}. Test: F1 Score: {f1:.2f}. Precision: {precision:.2f}, Recall: {recall:.2f}. {len(snapshot_sequence)} snapshots.')

def train(methods=['tabular','gnn'], use_saved_data=False, n_simulations=2, log_window=300, game_time= 700, max_start_time_step=400, graph_size='medium', random_cyber_agent_seed=None, number_of_epochs=10):

    # profiler = cProfile.Profile()
    # profiler.enable()

    snapshot_sequence = produce_training_data_parallel(use_saved_data=use_saved_data, 
                                                        n_simulations=n_simulations, 
                                                        log_window=log_window, 
                                                        game_time=game_time,
                                                        max_start_time_step=max_start_time_step, 
                                                        graph_size=graph_size, 
                                                        rddl_path='content/', 
                                                        random_cyber_agent_seed=random_cyber_agent_seed)

    create_masks(snapshot_sequence)

    logging.info(f'Number of snapshots: {len(snapshot_sequence)}')
    logging.info(f'Final snapshot (node type + log sequence, edge index, and labels):')
    logging.info(f'\n{snapshot_sequence[-1].x}')
    logging.info(snapshot_sequence[-1].edge_index)
    logging.info(snapshot_sequence[-1].y)

    # profiler.disable()

    # # Write the report to a file
    # with open('profiling_report.txt', 'w') as file:
    #     # Create a Stats object with the specified output stream
    #     stats = pstats.Stats(profiler, stream=file)
    #     stats.sort_stats('cumtime')
    #     stats.print_stats()
    # print("Profiling report saved to 'profiling_report.txt'")    

    if 'tabular' in methods:
        test_true_labels, test_predicted_labels = train_tabular(snapshot_sequence=snapshot_sequence, graph_size=graph_size)
        print_results('Tabular', snapshot_sequence, test_true_labels, test_predicted_labels)
    if 'gnn' in methods:
        test_true_labels, test_predicted_labels = train_gnn(number_of_epochs=number_of_epochs, snapshot_sequence=snapshot_sequence)
        print_results('GNN', snapshot_sequence, test_true_labels, test_predicted_labels)
    

