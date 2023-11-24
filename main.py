import logging
from trainer import train

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

for n_simulations in [32]:
    for log_window in [3]:
        train(methods=['tabular', 'gnn'],
            use_saved_data=False, 
            n_simulations=n_simulations, 
            log_window=log_window, 
            game_time= 70, 
            max_start_time_step=40, 
            graph_size='small', 
            random_cyber_agent_seed=0, 
            learning_rate=0.01,
            hidden_layers_list=[[32], [64], [128], [16, 16]],
            number_of_epochs=8)
