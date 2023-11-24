from trainer import train

for n_simulations in [4, 8, 16, 32]:
    for log_window in [3]:
        train(methods=['tabular', 'gnn'],
            use_saved_data=False, 
            n_simulations=n_simulations, 
            log_window=log_window, 
            game_time= 70, 
            max_start_time_step=40, 
            graph_size='small', 
            random_cyber_agent_seed=None, 
            number_of_epochs=10)
