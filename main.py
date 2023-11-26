import logging
from trainer import train

logging.basicConfig(filename='log.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.warning('\n\n')

# game_time=70 # Good for the small graph
game_time=500 # Good for the medium-sized graph
for n_simulations in [32]:
    for log_window in [16, 64]:
        train(methods=['gnn'], # ['tabular', 'gnn']
            use_saved_data=False, 
            n_simulations=n_simulations, 
            log_window=log_window, 
            game_time= game_time, 
            max_start_time_step=log_window + int((game_time-log_window)/2), 
            max_log_steps_after_total_compromise=int(log_window/2),
            graph_size='medium', 
            random_cyber_agent_seed=None, 
            batch_size=256,
            learning_rate_list=[0.005],
            hidden_layers_list=[[128, 128]],
            number_of_epochs=8)
