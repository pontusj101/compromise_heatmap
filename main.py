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


# Instance creation
parser.add_argument('--n_instances', type=int, default=16, help='Number of instances to create')
parser.add_argument('--min_size', type=int, default=16, help='Minimum number of hosts in each instance')
parser.add_argument('--max_size', type=int, default=16, help='Maximum number of hosts in each instance')
parser.add_argument('--n_init_compromised', type=int, default=4, help='Number of hosts initially compromised in each instance')
parser.add_argument('--extra_host_host_connection_ratio', type=float, default=0.25, help='0.25 means that 25% of hosts will have more than one connection to another host.')
parser.add_argument('--game_time', type=int, default=128, help='Max time horizon for the simulation. Will stop early if whole graph is compromised.') # small: 70, large: 500

# Simulation
parser.add_argument('-l', '--log_window', type=int, default=64, help='Size of the logging window')
parser.add_argument('--random_cyber_agent_seed', default=None, help='Seed for random cyber agent')
# and --rddl_path

# Training
parser.add_argument('--gnn_type', default='GAT', choices=['GAT', 'RGCN', 'GIN', 'GCN'], help='Type of GNN to use for training')
parser.add_argument('--max_instances', type=int, default=9999, help='Maximum number of instances to use for training')
parser.add_argument('--max_log_window', type=int, default=9999, help='Size of the logging window')
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for GNN training')
parser.add_argument('--n_hidden_layer_1', type=int, default=128, help='Number of neurons in hidden layer 1 for GNN')
parser.add_argument('--n_hidden_layer_2', type=int, default=128, help='Number of neurons in hidden layer 2 for GNN')
parser.add_argument('--n_hidden_layer_3', type=int, default=0, help='Number of neurons in hidden layer 3 for GNN')
parser.add_argument('--n_hidden_layer_4', type=int, default=0, help='Number of neurons in hidden layer 4 for GNN')
parser.add_argument('--edge_embedding_dim', type=int, default=16, help='Edge embedding dimension for GAT')
parser.add_argument('--heads_per_layer', type=int, default=2, help='Number of attention heads per layer for GAT')
parser.add_argument('--checkpoint_file', type=str, default=None, help='Name of the checkpoint file to resume training from.')

# Evaluation
parser.add_argument('--trigger_threshold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')
parser.add_argument('--evaluation_sequences', type=int, default=64, help='Frames per second in the animation.')
# and --predictor_filename and --predictor_type

# Animation
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

max_start_time_step = args.log_window + int((args.game_time - args.log_window) / 2)
max_log_steps_after_total_compromise = int(args.log_window / 2)
if args.log_window == -1:
    log_window = int(2.5 * args.max_size)
else:
    log_window = args.log_window
bucket_name = 'gnn_rddl'
storage_client = storage.Client()
bucket = storage_client.get_bucket(bucket_name)
config_blob = bucket.blob('config.json')
json_data = config_blob.download_as_string()
config = json.loads(json_data.decode('utf-8'))

# with open(CONFIG_FILE, 'r') as f:
#     config = json.load(f)

# Modes
if 'all' in args.modes:
    args.modes = ['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'eval', 'anim']

if 'instance' in args.modes:
    logging.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_instances,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        extra_host_host_connection_ratio=args.extra_host_host_connection_ratio,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')

if 'simulate' in args.modes:
    logging.info(f'Producing training data.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        snapshot_sequence_path=config['training_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'Training data produced and written to {config["training_sequence_dirpath"]}.')

if 'train' in args.modes:
    logging.info(f'Training GNN on a specific graph.')
    predictor_filename = train_gnn(
                    gnn_type=args.gnn_type,
                    sequence_file_name=config['training_sequence_dirpath'], 
                    max_instances=args.max_instances,
                    max_log_window=args.max_log_window,
                    number_of_epochs=args.epochs, 
                    learning_rate=args.learning_rate, 
                    batch_size=args.batch_size, 
                    n_hidden_layer_1=args.n_hidden_layer_1,
                    n_hidden_layer_2=args.n_hidden_layer_2,
                    n_hidden_layer_3=args.n_hidden_layer_3,
                    n_hidden_layer_4=args.n_hidden_layer_4,
                    edge_embedding_dim=args.edge_embedding_dim,
                    heads_per_layer=args.heads_per_layer, 
                    checkpoint_file=args.checkpoint_file)  # Add checkpoint file parameter

    config['predictor_filename'] = predictor_filename
    config['predictor_type'] = 'gnn'
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'GNN trained. Model written to {predictor_filename}.')

if 'eval_seq' in args.modes:
    logging.info(f'Producing {args.evaluation_sequences} evaluation snapshot sequences.')
    logging.info(f'Creating {args.evaluation_sequences} new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=args.evaluation_sequences,
        min_size=args.min_size,
        max_size=args.max_size,
        n_init_compromised=args.n_init_compromised,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        snapshot_sequence_path=config['evaluation_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'Evaulation data produced and written to {config["evaluation_sequence_dirpath"]}.')

if 'eval' in args.modes:
    evaluator = Evaluator(trigger_threshold=args.trigger_threshold)
    evaluator.evaluate_test_set(
        predictor_type=config['predictor_type'], 
        predictor_filename=config['predictor_filename'], 
        test_snapshot_sequence_path=config['evaluation_sequence_filepath'])

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
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    simulator.produce_training_data_parallel(
        bucket_name=bucket_name,
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        snapshot_sequence_path=config['animation_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    json_str = json.dumps(config, indent=4)
    config_blob.upload_from_string(json_str)
    logging.info(f'Animation data produced and written to {config["animation_sequence_dirpath"]}.')

if 'anim' in args.modes:
    logging.info(f'Creating animation.')
    animator = Animator(config['animation_sequence_filepath'], 
                        hide_prediction=args.hide_prediction, 
                        hide_state=args.hide_state)
    animator.create_animation(predictor_type=config['predictor_type'], 
                             predictor_filename=config['predictor_filename'],
                             frames_per_second=args.frames_per_second)
    s = f'Animation written to file network_animation.mp4.'
    logging.info(s)
    print(s)

if 'explore' in args.modes:
    logging.info(f'Exploring.')
    explorer = Explorer(config['predictor_type'], config['predictor_filename'])
    explorer.explore(config['evaluation_sequence_filepath'])

if 'clean' in args.modes:
    directories = ["/workspaces/rddl_training_data_producer/snapshot_sequences",
                   "/workspaces/rddl_training_data_producer/loss_curves",
                   "/workspaces/rddl_training_data_producer/tmp",
                   "/workspaces/rddl_training_data_producer/rddl"]
    for directory in directories:
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    directory = "/workspaces/rddl_training_data_producer/models"
    for filename in os.listdir(directory):
        if filename.startswith("model_n"):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)  # Remove the file
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    logging.info(f'Cleaned up.')
