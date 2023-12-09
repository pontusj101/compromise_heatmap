import argparse
import os
import logging
import warnings
import ast
import json
from animator import Animator
from gnn_explorer import Explorer
from instance_creator import create_instance
from simulator import Simulator
from evaluator import Evaluator
from gnn_trainer import train_gnn

# Constants
CONFIG_FILE = 'config.json'

# Initialize parser
parser = argparse.ArgumentParser(description='Run different modes of the security simulation program.')

# Adding arguments
parser.add_argument(
    'modes', 
    nargs='+',  # '+' means one or more arguments
    choices=['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'evaluate', 'animate', 'explore', 'all'], 
    help='Mode(s) of operation. Choose one or more from: instance, simulate, eval_seq, train, evaluate, animate, explore and all.'
)


# Instance creation
parser.add_argument('--n_instances', type=int, default=128, help='Number of instances to create')
parser.add_argument('--min_size', type=int, default=16, help='Minimum number of hosts in each instance')
parser.add_argument('--max_size', type=int, default=16, help='Maximum number of hosts in each instance')
parser.add_argument('--extra_host_host_connection_ratio', type=float, default=0.25, help='0.25 means that 25% of hosts will have more than one connection to another host.')
parser.add_argument('--game_time', type=int, default=2000, help='Max time horizon for the simulation. Will stop when whole graph is compromised.') # small: 70, large: 500

# Simulation
parser.add_argument('-l', '--log_window', type=int, default=128, help='Size of the logging window')
parser.add_argument('--random_cyber_agent_seed', default=None, help='Seed for random cyber agent')
# and --rddl_path

# Training
parser.add_argument('--max_instances', type=int, default=[128], help='Maximum number of instances to use for training')
parser.add_argument('--epochs', type=int, default=8, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for GNN training')
parser.add_argument('--hidden_layers', nargs='+', type=str, default="[[128, 128]]", help='Hidden layers configuration for GNN')

# Evaluation
parser.add_argument('--trigger_threshold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')
# and --predictor_filename and --predictor_type

# Animation
parser.add_argument('--frames_per_second', type=int, default=25, help='Frames per second in the animation.')
parser.add_argument('--hide_prediction', action='store_true', help='Hide prediction in the animation.')
parser.add_argument('--hide_state', action='store_true', help='Hide the attacker\'s state in the animation.')
# and --predictor_filename and --predictor_type

# Parse arguments
args = parser.parse_args()
hidden_layers_str = args.hidden_layers
# hidden_layers_str = hidden_layers_str.strip("[]'\"")
hidden_layers = ast.literal_eval(hidden_layers_str)

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

with open(CONFIG_FILE, 'r') as f:
    config = json.load(f)

# Modes
if 'all' in args.modes:
    args.modes = ['instance', 'simulate', 'eval_seq', 'anim_seq', 'train', 'evaluate', 'animate', 'clean']

if 'instance' in args.modes:
    logging.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=args.n_instances,
        min_size=args.min_size,
        max_size=args.max_size,
        extra_host_host_connection_ratio=args.extra_host_host_connection_ratio,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')

if 'simulate' in args.modes:
    logging.info(f'Producing training data.')
    simulator = Simulator()
    training_sequence_filepath = simulator.produce_training_data_parallel(
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        tmp_path=config['tmp_dirpath'],
        snapshot_sequence_path=config['snapshot_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    config['training_sequence_filepath'] = training_sequence_filepath
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'Training data produced and written to {training_sequence_filepath}.')


if 'eval_seq' in args.modes:
    logging.info(f'Producing eight evaluation snapshot sequences.')
    logging.info(f'Creating eight new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=32,
        min_size=args.min_size,
        max_size=args.max_size,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    animation_sequence_filepath = simulator.produce_training_data_parallel(
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        tmp_path=config['tmp_dirpath'],
        snapshot_sequence_path=config['snapshot_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    config['evaluation_sequence_filepath'] = animation_sequence_filepath
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'Evaulation data produced and written to {animation_sequence_filepath}.')

if 'anim_seq' in args.modes:
    logging.info(f'Producing single animantion snapshot sequence.')
    logging.info(f'Creating new instance specification.')
    instance_rddl_filepaths, graph_index_filepaths = create_instance( 
        rddl_path=config['rddl_dirpath'],
        n_instances=1,
        min_size=args.min_size,
        max_size=args.max_size,
        horizon=args.game_time)
    config['instance_rddl_filepaths'] = instance_rddl_filepaths
    config['graph_index_filepaths'] = graph_index_filepaths
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'{len(instance_rddl_filepaths)} instance specifications and graph indicies written to file.')
    simulator = Simulator()
    animation_sequence_filepath = simulator.produce_training_data_parallel(
        domain_rddl_path=config['domain_rddl_filepath'],
        instance_rddl_filepaths=config['instance_rddl_filepaths'],
        graph_index_filepaths=config['graph_index_filepaths'],
        rddl_path=config['rddl_dirpath'], 
        tmp_path=config['tmp_dirpath'],
        snapshot_sequence_path=config['snapshot_sequence_dirpath'],
        log_window=log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    config['animation_sequence_filepath'] = animation_sequence_filepath
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'Animation data produced and written to {animation_sequence_filepath}.')

if 'train' in args.modes:
    logging.info(f'Training GNN on a specific graph.')
    predictor_filename = train_gnn(
                    sequence_file_name=config['training_sequence_filepath'], 
                    max_instances_list=args.max_instances,
                    number_of_epochs=args.epochs, 
                    learning_rate=args.learning_rate, 
                    batch_size=args.batch_size, 
                    hidden_layers_list=hidden_layers)
    config['predictor_filename'] = predictor_filename
    config['predictor_type'] = 'gnn'
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logging.info(f'GNN trained. Model written to {predictor_filename}.')


if 'evaluate' in args.modes:
    evaluator = Evaluator(trigger_threshold=args.trigger_threshold)
    evaluator.evaluate_test_set(
        predictor_type=config['predictor_type'], 
        predictor_filename=config['predictor_filename'], 
        test_snapshot_sequence_path=config['evaluation_sequence_filepath'])

if 'animate' in args.modes:
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
