from trainer import train

train(method='gnn', use_saved_data=False, n_simulations=2, log_window=300, game_time= 700, max_start_time_step=400, graph_size='medium', random_cyber_agent_seed=None, number_of_epochs=3)
