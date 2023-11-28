import logging
from animator import create_animation
from instance_creator import create_instance
from simulator import produce_training_data_parallel
from gnn_trainer import train_gnn

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.warning('\n\n')

mode = 'animate' # 'create_instance', 'produce_training_data', 'train_gnn', 'animate', 'evaulate'
instance_type = 'static'
size = 'small'
game_time=70 # horizon = 70 good for the small graph, 500 good for the medium-sized graph, 1000 good for the large graph
rddl_path='content/'
tmp_path='tmp/'
snapshot_sequence_path = 'snapshot_sequences/'
training_sequence_file_name = 'snapshot_sequences/latest20231127_192822.pkl'
animation_sequence_filename = 'snapshot_sequences/latest20231127_191906.pkl'
animation_predictor_filename='models/model_hl_[64, 64]_n_6287_lr_0.005_bs_256.pt'
n_simulations = 128
log_window = 3
random_cyber_agent_seed=None
number_of_epochs = 8
learning_rate=0.005
batch_size = 256
hidden_layers=[64, 64]
animation_predictor_type='gnn'

max_start_time_step = log_window + int((game_time-log_window)/2)
max_log_steps_after_total_compromise = int(log_window/2)

if mode == 'create_instance':
    # TODO: Write graph_index to file 
    logging.info(f'Creating new instance specification.')
    rddl_file_path, graph_index_file_path = create_instance(instance_type=instance_type, size=size, horizon=game_time, rddl_path=rddl_path)
    logging.info(f'Instance specification written to {rddl_file_path}. Graph index written to {graph_index_file_path}.')

if mode == 'produce_training_data':
    logging.info(f'Producing training data.')
    sequence_file_name = produce_training_data_parallel(
        n_simulations=n_simulations, 
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        rddl_path=rddl_path, 
        tmp_path=tmp_path,
        snapshot_sequence_path=snapshot_sequence_path,
        random_cyber_agent_seed=random_cyber_agent_seed)
    logging.info(f'Training data produced and written to {sequence_file_name}.')

if mode == 'train_gnn':
    logging.info(f'Training GNN.')
    test_true_labels, test_predicted_labels, predictor_filename = train_gnn(
                    number_of_epochs=number_of_epochs, 
                    sequence_file_name=training_sequence_file_name, 
                    learning_rate=learning_rate, 
                    batch_size=batch_size, 
                    hidden_layers=hidden_layers)
    logging.info(f'GNN trained. Model written to {predictor_filename}.')

if mode == 'animate':
    logging.info(f'Creating animation.')
    create_animation(animation_sequence_filename=animation_sequence_filename, 
                     predictor_type=animation_predictor_type, 
                     predictor_filename = animation_predictor_filename)
    logging.info(f'Animation created.')

if 'evaulate' in mode:
    pass