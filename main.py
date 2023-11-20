from gnn_trainer import train_model

train_model(use_saved_data=False, n_simulations=2, log_window=50, game_time= 250, max_start_time_step=100, graph_size='medium', number_of_epochs=10, debug_print=1)