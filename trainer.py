import logging
from simulator import produce_training_data_parallel
from gnn_trainer import train_gnn
from tabular_trainer import train_tabular

def train(method='gnn', use_saved_data=False, n_simulations=2, log_window=300, game_time= 700, max_start_time_step=400, graph_size='medium', random_cyber_agent_seed=None, number_of_epochs=10):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

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

    logging.info(f'Number of snapshots: {len(snapshot_sequence)}')
    logging.info(f'Final snapshot (node type + log sequence, edge index, and labels):')
    logging.info(snapshot_sequence[-1].x)
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

    if method == 'gnn':
        train_gnn(number_of_epochs=number_of_epochs, snapshot_sequence=snapshot_sequence)
    elif method == 'tabular':
        train_tabular(snapshot_sequence=snapshot_sequence)