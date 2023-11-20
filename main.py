from gnn_trainer import train_model

train_model(use_saved_data=False, n_simulations=8, log_window=300, game_time= 700, max_start_time_step=400, graph_size='medium', random_cyber_agent_seed=None, number_of_epochs=10, debug_print=1)