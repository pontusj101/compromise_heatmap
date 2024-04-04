import argparse
import os
import warnings
import ast
import json
from google.cloud import storage
from google.cloud import secretmanager
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from .animator import Animator
from .gnn_explorer import Explorer
from .instance_creator import create_instance
from .simulator import Simulator
from .evaluator import Evaluator
from .gnn_trainer import train_gnn
from .bucket_manager import BucketManager
from . import logging as heatmap_logging

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'auth.json'

# Constants
CONFIG_FILE = 'config.json'

# Initialize parser
parser = argparse.ArgumentParser(description='Run different modes of the security simulation program.')

# Adding arguments
parser.add_argument(
    'modes',
    nargs='+',  # '+' means one or more arguments
    choices=['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'eval', 'anim', 'explore', 'clean', 'all'],
    help='Mode(s) of operation. Choose one or more from: instance, simulate, eval_seq, train, eval, anim, explore, clean and all.'
)
parser.add_argument('--debug', action="store_true")
parser.add_argument('--bucket_name', type=str, default='gnn_rddl', help='Name of the GCP bucket to use for storage.')

# Instance creation and training
parser.add_argument('--min_size', type=int, default=72, help='Minimum number of hosts in each instance')
parser.add_argument('--max_size', type=int, default=72, help='Maximum number of hosts in each instance')
parser.add_argument('--min_game_time', type=int, default=8, help='Min time horizon for the simulation and training.') # small: 70, large: 500
parser.add_argument('--max_game_time', type=int, default=1024, help='Max time horizon for the simulation and training. Will stop simulation early if whole graph is compromised.') # small: 70, large: 500

# Instance creation
parser.add_argument('--n_instances', type=int, default=256, help='Number of instances to create')
parser.add_argument('--random_initial_compromise', action='store_true', help='Randomly choose all initial compromised hosts.')
parser.add_argument('--n_init_compromised', type=int, default=3, help='Number of hosts initially compromised in each instance')
parser.add_argument('--extra_host_host_connection_ratio', type=float, default=0.25, help='0.25 means that 25% of hosts will have more than one connection to another host.')

# Simulation
parser.add_argument('-l', '--sim_log_window', type=int, default=64, help='Size of the logging window')
parser.add_argument('--agent_type', default='passive', choices=['random', 'less_random', 'host_targeted', 'novelty', 'keyboard', 'passive'], help='Type of agent to use for simulation')
parser.add_argument('--novelty_priority', type=int, default=2, help='Priority of newly discovered actions for less_random agent')
parser.add_argument('--random_agent_seed', default=None, help='Seed for random cyber agent')

# Training
parser.add_argument('--enable_wandb', action='store_true')
parser.add_argument('--gnn_type', default='GAT', choices=['GAT', 'RGCN', 'GIN', 'GCN', 'GAT_LSTM'], help='Type of GNN to use for training')
parser.add_argument('--max_training_sequences', type=int, default=128, help='Maximum number of instances to use for training')
parser.add_argument('--n_validation_sequences', type=int, default=32, help='Number of sequences to use for validation')
parser.add_argument('--n_uncompromised_sequences', type=int, default=64, help='Number of uncompromised sequences to use')
parser.add_argument('--train_log_window', type=int, default=64, help='Size of the logging window')
parser.add_argument('--epochs', type=int, default=32, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for GNN training')
parser.add_argument('--n_hidden_layer_1', type=int, default=128, help='Number of neurons in hidden layer 1 for GNN')
parser.add_argument('--n_hidden_layer_2', type=int, default=128, help='Number of neurons in hidden layer 2 for GNN')
parser.add_argument('--n_hidden_layer_3', type=int, default=128, help='Number of neurons in hidden layer 3 for GNN')
parser.add_argument('--n_hidden_layer_4', type=int, default=0, help='Number of neurons in hidden layer 4 for GNN')
parser.add_argument('--edge_embedding_dim', type=int, default=128, help='Edge embedding dimension for GAT')
parser.add_argument('--heads_per_layer', type=int, default=2, help='Number of attention heads per layer for GAT')
parser.add_argument('--lstm_hidden_dim', type=int, default=128, help='Number of neurons in LSTM hidden layer for GNN_LSTM')
parser.add_argument('--checkpoint_file', type=str, default=None, help='Name of the checkpoint file to resume training from.')

# Evaluation and animation
# parser.add_argument('--model_filepath', type=str, default='models/model_log_window__hl_16,256,256_nsnpsht_256_lr_0.0002_bs_128_20231228_121849.pt', help='Path the model filename, relative to the bucket root.')
parser.add_argument('--model_filepath', type=str, default='models/model/GAT/log_window__hl_380,209,439_lstm_284_nsnpsht_128_lr_0.0002_bs_128_20231231_143651.pt', help='Path the model filename, relative to the bucket root.')
# LSTM parser.add_argument('--model_filepath', type=str, default='models/model/GAT_LSTM/log_window__hl_256,256,256_lstm_256_nsnpsht_256_lr_0.0002_bs_128_20231230_172202.pt', help='Path the model filename, relative to the bucket root.')
# Evaluation
parser.add_argument('--trigger_threshold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')
parser.add_argument('--n_evaluation_sequences', type=int, default=32, help='Number of evaluation sequences to create')

# and --predictor_filename and --predictor_type

# Animation
# parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_255/252_nodes/257_snapshots/20231215_134213_9153.pkl', help='Path the animation sequence filename, relative to the bucket root.')
# passive parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_1/32_nodes/255_snapshots/passive/20231228_135631_9990.pkl', help='Path the animation sequence filename, relative to the bucket root.')
# 32 large? parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_64/32_nodes/150_snapshots/random/20231228_204628_5684.pkl', help='Path the animation sequence filename, relative to the bucket root.')
parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_64/256_nodes/960_snapshots/novelty/20231231_210248_3828.pkl', help='Path the animation sequence filename, relative to the bucket root.')
parser.add_argument('--frames_per_second', type=int, default=25, help='Frames per second in the animation.')
parser.add_argument('--n_init_compromised_animate', type=int, default=1, help='Number of hosts initially compromised in each instance')
parser.add_argument('--hide_prediction', action='store_true', help='Hide prediction in the animation.')
parser.add_argument('--hide_state', action='store_true', help='Hide the attacker\'s state in the animation.')
# and --predictor_filename and --predictor_type

# Parse arguments
args = parser.parse_args()

heatmap_logging.debug = args.debug
logger = heatmap_logging.setup_logging()

warnings.filterwarnings("ignore", message="Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations", module="pyRDDLGym.Visualizer.ChartViz")
warnings.filterwarnings("ignore", message="Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.", module="pyRDDLGym.Visualizer.ChartViz")
warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims makes transformation singular; automatically expanding.", module="pyRDDLGym.Visualizer.ChartViz")

max_start_time_step = args.sim_log_window + int((args.max_game_time - args.sim_log_window) / 2)
max_log_steps_after_total_compromise = int(args.sim_log_window / 2)
if args.sim_log_window == -1:
    sim_log_window = int(2.5 * args.max_size)
else:
    sim_log_window = args.sim_log_window

bucket_manager = BucketManager(args.bucket_name)
config = bucket_manager.load_config_file(CONFIG_FILE)

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Modes
if 'all' in args.modes:
    args.modes = ['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'eval', 'anim']

if 'instance' in args.modes:
    logger.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance(
        bucket_name=args.bucket_name,
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_instances,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        random_inital_compromise=args.random_initial_compromise,
        extra_host_host_connection_ratio=args.extra_host_host_connection_ratio,
        horizon=args.max_game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    bucket_manager.json_save_to_bucket(config, CONFIG_FILE)
    logger.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')

if 'simulate' in args.modes:
    logger.info(f'Producing training data.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=args.bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        rddl_path=config['rddl_dirpath'],
        snapshot_sequence_path=config['training_sequence_dirpath'],
        log_window=sim_log_window,
        max_start_time_step=max_start_time_step,
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        agent_type=args.agent_type,
        novelty_priority=args.novelty_priority,
        random_agent_seed=args.random_agent_seed)
    logger.info(f'Training data produced and written to {config["training_sequence_dirpath"]}.')

if 'train' in args.modes:
    logger.info(f'Training {args.gnn_type}.')


    wandb_api_key = None
    if args.enable_wandb:
        # smsClient = secretmanager.SecretManagerServiceClient()
        # name = 'projects/473095460232/secrets/Weights_Biases_API_Key/versions/latest'
        # response = smsClient.access_secret_version(request={"name": name})
        # wandb_api_key = response.payload.data.decode("UTF-8")
        with open('./wandb.key') as f:
            wandb_api_key = read(f)

    predictor_filename = train_gnn(
                    wandb_api_key=wandb_api_key,
                    gnn_type=args.gnn_type,
                    bucket_manager=bucket_manager,
                    sequence_dir_path=config['training_sequence_dirpath'],
                    model_dirpath='models/',
                    number_of_epochs=args.epochs,
                    max_training_sequences=args.max_training_sequences,
                    n_validation_sequences=args.n_validation_sequences,
                    n_uncompromised_sequences=args.n_uncompromised_sequences,
                    min_nodes=args.min_size,
                    max_nodes=args.max_size,
                    min_snapshots=args.min_game_time,
                    max_snapshots=args.max_game_time,
                    log_window=args.train_log_window,
                    learning_rate=args.learning_rate,
                    batch_size=args.batch_size,
                    n_hidden_layer_1=args.n_hidden_layer_1,
                    n_hidden_layer_2=args.n_hidden_layer_2,
                    n_hidden_layer_3=args.n_hidden_layer_3,
                    n_hidden_layer_4=args.n_hidden_layer_4,
                    edge_embedding_dim=args.edge_embedding_dim,
                    heads_per_layer=args.heads_per_layer,
                    lstm_hidden_dim=args.lstm_hidden_dim,
                    checkpoint_interval=1,  # Add a parameter to set checkpoint interval
                    checkpoint_file=args.checkpoint_file,  # Add checkpoint file parameter
                    checkpoint_path='checkpoints/')

    logger.info(f'{args.gnn_type} trained. Model written to {predictor_filename}.')

if 'eval_seq' in args.modes:
    logger.info(f'Producing {args.n_evaluation_sequences} evaluation snapshot sequences.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance(
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_evaluation_sequences,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        random_inital_compromise=args.random_initial_compromise,
        horizon=args.max_game_time)
    config['eval_instance_rddl_filepaths'] = instance_rddl_filepaths
    config['eval_graph_index_filepaths'] = graph_index_filepaths
    bucket_manager.json_save_to_bucket(config, CONFIG_FILE)
    logger.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=args.bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=instance_rddl_filepaths,
        graph_index_filepaths=graph_index_filepaths,
        rddl_path=config['rddl_dirpath'],
        snapshot_sequence_path=config['evaluation_sequence_dirpath'],
        log_window=sim_log_window,
        max_start_time_step=max_start_time_step,
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        agent_type=args.agent_type,
        novelty_priority=args.novelty_priority,
        random_agent_seed=args.random_agent_seed)
    logger.info(f'Evaulation data produced and written to {config["evaluation_sequence_dirpath"]}.')

if 'eval' in args.modes:
    evaluator = Evaluator(trigger_threshold=args.trigger_threshold)
    evaluator.evaluate_test_set(
        predictor_type=config['predictor_type'],
        predictor_filename=args.model_filepath,
        bucket_manager=bucket_manager,
        sequence_dir_path = config['evaluation_sequence_dirpath'],
        min_nodes = args.min_size,
        max_nodes = args.max_size,
        log_window = args.train_log_window,
        max_sequences = args.n_evaluation_sequences)

if 'anim_seq' in args.modes:
    logger.info(f'Producing single animantion snapshot sequence.')
    logger.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance(
        rddl_path=config['rddl_dirpath'],
        n_instances=1,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised_animate,
        random_inital_compromise=args.random_initial_compromise,
        horizon=args.max_game_time)
    logger.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=args.bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=instance_rddl_filepaths,
        graph_index_filepaths=graph_index_filepaths,
        rddl_path=config['rddl_dirpath'],
        snapshot_sequence_path=config['animation_sequence_dirpath'],
        log_window=sim_log_window,
        max_start_time_step=max_start_time_step,
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        agent_type=args.agent_type,
        novelty_priority=args.novelty_priority,
        random_agent_seed=args.random_agent_seed)
    logger.info(f'Animation data produced and written to {config["animation_sequence_dirpath"]}.')

if 'anim' in args.modes:
    logger.info(f'Creating animation.')

    animator = Animator(config['domain_rddl_filepath'],
                        args.animation_sequence_filepath,
                        bucket_manager=bucket_manager,
                        hide_prediction=args.hide_prediction,
                        hide_state=args.hide_state)
    animator.create_animation(predictor_type=config['predictor_type'],
                             predictor_filename=config['predictor_filename'],
                             frames_per_second=args.frames_per_second)
    s = f'Animation written to file.'
    logger.info(s)
    print(s)


