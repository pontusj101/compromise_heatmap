import logging
from trainer import train
from animator import animate_snapshot_sequence
from instance_creator import create_instance

logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.warning('\n\n')

mode = 'both' # 'train' or 'animate' or 'both'
model_file_name = 'models/model_hl_[256, 256]_n_421765_lr_0.005_bs_256.pt'

logging.info(f'Creating new instance specification.')
game_time=70 # Good for the small graph
# game_time=500 # Good for the medium-sized graph
# game_time = 1000 # Good for the large graph
log_window = 3
max_start_time_step = log_window + int((game_time-log_window)/2)
max_log_steps_after_total_compromise = int(log_window/2),

graph_index = create_instance(instance_type='static', size='small', horizon=game_time, rddl_path='content/')

if mode == 'train' or mode == 'both':
    for n_simulations in [128]:
        model_file_name = train(methods=['tabular'], # ['tabular', 'gnn']
            use_saved_data=False, 
            n_simulations=n_simulations, 
            log_window=log_window, 
            game_time= game_time, 
            max_start_time_step=max_start_time_step, 
            max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
            graph_index=graph_index,
            random_cyber_agent_seed=None, 
            batch_size=256,
            learning_rate_list=[0.005],
            hidden_layers_list=[[64, 64]],
            number_of_epochs=8,
            evaluate=False)
if mode == 'animate' or mode == 'both':
    animate_snapshot_sequence(predictor_type='tabular', 
                              predictor_filename=model_file_name, 
                              graph_index=graph_index, 
                              game_time=game_time, 
                              max_start_time_step=max_start_time_step, 
                              max_log_steps_after_total_compromise=max_log_steps_after_total_compromise, 
                              log_window=log_window)