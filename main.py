import logging
from animator import animate_snapshot_sequence
from instance_creator import create_instance
from simulator import produce_training_data_parallel
from gnn_trainer import train_gnn

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.warning('\n\n')

mode = ['train_gnn', 'animate'] # 'train_gnn', 'animate', 'evaulate'
predictor_file_name = 'models/model_hl_[256, 256]_n_421765_lr_0.005_bs_256.pt'
game_time=70 # 70 good for the small graph, 500 good for the medium-sized graph, 1000 good for the large graph
log_window = 3
max_start_time_step = log_window + int((game_time-log_window)/2)
max_log_steps_after_total_compromise = int(log_window/2)

logging.info(f'Creating new instance specification.')
graph_index = create_instance(instance_type='static', size='small', horizon=game_time, rddl_path='content/')

logging.info(f'Producing training data.')
sequence_file_name = produce_training_data_parallel(
    use_saved_data=False, 
    n_simulations=8, 
    log_window=log_window, 
    game_time=game_time,
    max_start_time_step=max_start_time_step, 
    max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
    graph_index=graph_index,
    rddl_path='content/', 
    tmp_path='tmp/',
    snapshot_sequence_path = 'snapshot_sequences/',
    random_cyber_agent_seed=None)

predictor_file_name = sequence_file_name
if 'train_gnn' in mode:
    test_true_labels, test_predicted_labels, predictor_file_name = train_gnn(
                    number_of_epochs=8, 
                    sequence_file_name=sequence_file_name, 
                    learning_rate=0.005, 
                    batch_size=256, 
                    hidden_layers=[64, 64])

if 'animate' in mode:
    animate_snapshot_sequence(predictor_type='gnn', 
                              predictor_filename=predictor_file_name, 
                              graph_index=graph_index, 
                              game_time=game_time, 
                              max_start_time_step=max_start_time_step, 
                              max_log_steps_after_total_compromise=max_log_steps_after_total_compromise, 
                              log_window=log_window)

if 'evaulate' in mode:
    pass