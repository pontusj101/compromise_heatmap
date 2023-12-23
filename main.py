import argparse
import os
import logging
import warnings
import ast
import json
from google.cloud import storage
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler
from animator import Animator
from gnn_explorer import Explorer
from instance_creator import create_instance
from simulator import Simulator
from evaluator import Evaluator
from gnn_trainer import train_gnn
from bucket_manager import BucketManager

if os.environ.get('CODESPACES') == 'true':
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'gnn-rddl-b602eb3e4b45.json'

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
parser.add_argument('--bucket_name', type=str, default='gnn_rddl', help='Name of the GCP bucket to use for storage.')

# Instance creation and training
parser.add_argument('--min_size', type=int, default=16, help='Minimum number of hosts in each instance')
parser.add_argument('--max_size', type=int, default=32, help='Maximum number of hosts in each instance')

# Instance creation
parser.add_argument('--n_instances', type=int, default=32, help='Number of instances to create')
parser.add_argument('--n_init_compromised', type=int, default=1, help='Number of hosts initially compromised in each instance')
parser.add_argument('--extra_host_host_connection_ratio', type=float, default=0.25, help='0.25 means that 25% of hosts will have more than one connection to another host.')
parser.add_argument('--game_time', type=int, default=256, help='Max time horizon for the simulation. Will stop early if whole graph is compromised.') # small: 70, large: 500

# Simulation
parser.add_argument('-l', '--sim_log_window', type=int, default=64, help='Size of the logging window')
parser.add_argument('--agent_type', default='random', choices=['random', 'host_targeted', 'keyboard', 'passive'], help='Type of agent to use for simulation')
parser.add_argument('--random_agent_seed', default=None, help='Seed for random cyber agent')
# and --rddl_path

# Training
parser.add_argument('--gnn_type', default='GAT_LSTM', choices=['GAT', 'RGCN', 'GIN', 'GCN', 'GAT_LSTM'], help='Type of GNN to use for training')
parser.add_argument('--max_training_sequences', type=int, default=128, help='Maximum number of instances to use for training')
parser.add_argument('--n_validation_sequences', type=int, default=64, help='Number of sequences to use for validation')
parser.add_argument('--train_log_window', type=int, default=64, help='Size of the logging window')
parser.add_argument('--epochs', type=int, default=2, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for GNN training')
parser.add_argument('--n_hidden_layer_1', type=int, default=1024, help='Number of neurons in hidden layer 1 for GNN')
parser.add_argument('--n_hidden_layer_2', type=int, default=512, help='Number of neurons in hidden layer 2 for GNN')
parser.add_argument('--n_hidden_layer_3', type=int, default=256, help='Number of neurons in hidden layer 3 for GNN')
parser.add_argument('--n_hidden_layer_4', type=int, default=0, help='Number of neurons in hidden layer 4 for GNN')
parser.add_argument('--edge_embedding_dim', type=int, default=128, help='Edge embedding dimension for GAT')
parser.add_argument('--heads_per_layer', type=int, default=4, help='Number of attention heads per layer for GAT')
parser.add_argument('--lstm_hidden_dim', type=int, default=128, help='Number of neurons in LSTM hidden layer for GNN_LSTM')
parser.add_argument('--checkpoint_file', type=str, default=None, help='Name of the checkpoint file to resume training from.')

# Evaluation and animation
# Best model so far
# parser.add_argument('--model_filepath', type=str, default='models/model_ninst_512_lw_256_nnodes_28_gt_512_20231214_102112_4614_hl_1448,2896,2896_nsnpsht_32526_lr_0.0010000000000000002_bs_91_20231216_070612.pt', help='Path the model filename, relative to the bucket root.')
parser.add_argument('--model_filepath', type=str, default='models/model_log_window_64/_hl_8192,8192,286_nsnpsht_64_lr_0.0006_bs_35_20231218_153719.pt', help='Path the model filename, relative to the bucket root.')
# Some other model
# parser.add_argument('--model_filepath', type=str, default='models/model_ninst_512_lw_256_nnodes_28_gt_512_20231214_102112_4614_hl_569,1532,5172_nsnpsht_16945_lr_0.0010334153904374018_bs_45_20231216_001205.pt', help='Path the model filename, relative to the bucket root.')
# Evaluation
parser.add_argument('--trigger_threshold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')
parser.add_argument('--n_evaluation_sequences', type=int, default=32, help='Number of evaluation sequences to create')

# and --predictor_filename and --predictor_type

# Animation
# parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_255/252_nodes/257_snapshots/20231215_134213_9153.pkl', help='Path the animation sequence filename, relative to the bucket root.')
parser.add_argument('--animation_sequence_filepath', type=str, default='animation_sequences/log_window_64/64_nodes/192_snapshots/20231218_163349_4471.pkl', help='Path the animation sequence filename, relative to the bucket root.')
parser.add_argument('--frames_per_second', type=int, default=25, help='Frames per second in the animation.')
parser.add_argument('--n_init_compromised_animate', type=int, default=1, help='Number of hosts initially compromised in each instance')
parser.add_argument('--hide_prediction', action='store_true', help='Hide prediction in the animation.')
parser.add_argument('--hide_state', action='store_true', help='Hide the attacker\'s state in the animation.')
# and --predictor_filename and --predictor_type

# Parse arguments
args = parser.parse_args()

client = google.cloud.logging.Client()
handler = CloudLoggingHandler(client)
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(handler)
# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Set the log level
# Clear existing handlers
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
# Create a file handler and set level to debug
file_handler = logging.FileHandler('log.log')
file_handler.setLevel(logging.DEBUG)
# Create a console (stream) handler and set level to debug
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create a formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
# Set formatter for file and console handlers
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add file and console handlers to the root logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# Supress unwanted logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations", module="pyRDDLGym.Visualizer.ChartViz")
warnings.filterwarnings("ignore", message="Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.", module="pyRDDLGym.Visualizer.ChartViz")
warnings.filterwarnings("ignore", message="Attempting to set identical low and high ylims makes transformation singular; automatically expanding.", module="pyRDDLGym.Visualizer.ChartViz")

max_start_time_step = args.sim_log_window + int((args.game_time - args.sim_log_window) / 2)
max_log_steps_after_total_compromise = int(args.sim_log_window / 2)
if args.sim_log_window == -1:
    sim_log_window = int(2.5 * args.max_size)
else:
    sim_log_window = args.sim_log_window

bucket_manager = BucketManager(args.bucket_name)
config = bucket_manager.json_load_from_bucket(CONFIG_FILE)

# with open(CONFIG_FILE, 'r') as f:
#     config = json.load(f)

# Modes
if 'all' in args.modes:
    args.modes = ['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'eval', 'anim']

if 'instance' in args.modes:
    logging.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        bucket_name=args.bucket_name,
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_instances,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        extra_host_host_connection_ratio=args.extra_host_host_connection_ratio,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    bucket_manager.json_save_to_bucket(config, CONFIG_FILE)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')

if 'simulate' in args.modes:
    logging.info(f'Producing training data.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=args.bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        snapshot_sequence_path=config['training_sequence_dirpath'],
        log_window=sim_log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        agent_type=args.agent_type,
        random_agent_seed=args.random_agent_seed)
    logging.info(f'Training data produced and written to {config["training_sequence_dirpath"]}.')

if 'train' in args.modes:
    logging.info(f'Training GNN on a specific graph.')
    predictor_filename = train_gnn(
                    gnn_type=args.gnn_type,
                    bucket_manager=bucket_manager,
                    sequence_dir_path=config['training_sequence_dirpath'],
                    model_dirpath='models/',
                    number_of_epochs=args.epochs, 
                    max_training_sequences=args.max_training_sequences,
                    n_validation_sequences=args.n_validation_sequences,
                    min_nodes=args.min_size,
                    max_nodes=args.max_size,
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


    logging.info(f'GNN trained. Model written to {predictor_filename}.')

if 'eval_seq' in args.modes:
    logging.info(f'Producing {args.n_evaluation_sequences} evaluation snapshot sequences.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_evaluation_sequences,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        horizon=args.game_time)
    config['eval_instance_rddl_filepaths'] = instance_rddl_filepaths
    config['eval_graph_index_filepaths'] = graph_index_filepaths
    bucket_manager.json_save_to_bucket(config, CONFIG_FILE)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
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
        random_agent_seed=args.random_agent_seed)
    logging.info(f'Evaulation data produced and written to {config["evaluation_sequence_dirpath"]}.')

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
    logging.info(f'Producing single animantion snapshot sequence.')
    logging.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=1,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised_animate,
        horizon=args.game_time)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
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
        random_agent_seed=args.random_agent_seed)
    logging.info(f'Animation data produced and written to {config["animation_sequence_dirpath"]}.')

if 'anim' in args.modes:
    logging.info(f'Creating animation.')

    animator = Animator(args.animation_sequence_filepath,
                        bucket_manager=bucket_manager,
                        hide_prediction=args.hide_prediction, 
                        hide_state=args.hide_state)
    animator.create_animation(predictor_type=config['predictor_type'], 
                             predictor_filename=args.model_filepath,
                             frames_per_second=args.frames_per_second)
    s = f'Animation written to file network_animation.mp4.'
    logging.info(s)
    print(s)


