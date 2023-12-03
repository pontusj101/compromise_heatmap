import argparse
import logging
import warnings
import ast
from animator import Animator
from instance_creator import create_instance
from simulator import Simulator
from evaluator import Evaluator
from gnn_trainer import train_gnn

# Initialize parser
parser = argparse.ArgumentParser(description='Run different modes of the security simulation program.')

# Adding arguments
parser.add_argument(
    'modes', 
    nargs='+',  # '+' means one or more arguments
    choices=['instance', 'simulate', 'eval_seq', 'train', 'evaluate', 'animate'], 
    help='Mode(s) of operation. Choose one or more from: instance, simulate, train, animate, evaluate.'
)


# Instance creation
parser.add_argument('--instance_type', default='random', choices=['static', 'random'], help='Type of instance to create')
parser.add_argument('--size', default='large', choices=['small', 'medium', 'large'], help='Size of the graph')
parser.add_argument('--game_time', type=int, default=2500, help='Time horizon for the simulation')
parser.add_argument('--rddl_path', default='rddl/', help='Path to the RDDL files')

# Simulation
parser.add_argument('-n', '--n_simulations', type=int, default=16, help='Number of simulations to run')
parser.add_argument('-l', '--log_window', type=int, default=8, help='Size of the logging window')
parser.add_argument('--domain_rddl_path', default='rddl/domain.rddl', help='Path to RDDL domain specification')
parser.add_argument('--instance_rddl_path', default='rddl/instance_random_large_2500_20231203_174808.rddl' , help='Path to RDDL instance specification')
parser.add_argument('--graph_index_path', default='rddl/graph_index_random_large_2500_20231203_174808.pkl', help='Path to pickled GraphIndex class.')
parser.add_argument('--tmp_path', default='tmp/', help='Temporary file path')
parser.add_argument('--snapshot_sequence_path', default='snapshot_sequences/', help='Path to snapshot sequences')
parser.add_argument('--random_cyber_agent_seed', default=None, help='Seed for random cyber agent')
# and --rddl_path

# Training
parser.add_argument('--epochs', type=int, default=4, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for GNN training')
parser.add_argument('--hidden_layers', nargs='+', type=str, default="[[128, 128]]", help='Hidden layers configuration for GNN')
parser.add_argument('--training_sequence_file_name', default='snapshot_sequences/snapshot_sequence_n16_l8_random_large_2500_20231203_174808.pkl', help='Filename for training sequence')

# Evaluation
parser.add_argument('--evaluation_sequence_path', default='snapshot_sequences/snapshot_sequence_n1_l8_random_large_2500_20231203_174808.pkl', help='Filename for evaulation sequence')
parser.add_argument('--trigger_threashold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--predictor_filename', default='models/model_n16_l8_random_large_2500_20231203_174808_hl_[128, 128]_n_29270_lr_0.001_bs_256.pt', help='Filename for the predictor model')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')
# and --predictor_filename and --predictor_type

# Animation
parser.add_argument('--animation_sequence_filename', default='snapshot_sequences/snapshot_sequence_n1_l8_random_large_2500_20231203_174808.pkl', help='Filename for animation sequence')
# and --predictor_filename and --predictor_type

# Parse arguments
args = parser.parse_args()

hidden_layers_str = args.hidden_layers
# hidden_layers_str = hidden_layers_str.strip("[]'\"")
hidden_layers = ast.literal_eval(hidden_layers_str)

# Set up logging
logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
warnings.filterwarnings("ignore", message="Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations", module="pyRDDLGym.Visualizer.ChartViz")
logging.warning('\n\n')

max_start_time_step = args.log_window + int((args.game_time - args.log_window) / 2)
max_log_steps_after_total_compromise = int(args.log_window / 2)

predictor_filename = None
instance_rddl_path = None
graph_index_path = None
training_sequence_path = None
evaluation_sequence_path = None

# Modes
if 'instance' in args.modes:
    # TODO: Write graph_index to file 
    logging.info(f'Creating new instance specification.')
    instance_rddl_path, graph_index_path = create_instance(
        instance_type=args.instance_type, 
        size=args.size, 
        horizon=args.game_time, 
        rddl_path=args.rddl_path)
    s = f'Instance specification written to {instance_rddl_path}. Graph index written to {graph_index_path}.'
    logging.info(s)
    print(s)

if 'simulate' in args.modes:
    if instance_rddl_path is None:
        instance_rddl_path = args.instance_rddl_path
    if graph_index_path is None:
        graph_index_path = args.graph_index_path
    else:
        logging.info(f'Using newly generated instance {instance_rddl_path} and graph index {graph_index_path}.')
    logging.info(f'Producing training data.')
    simulator = Simulator()
    training_sequence_path = simulator.produce_training_data_parallel(
        domain_rddl_path=args.domain_rddl_path,
        instance_rddl_path=instance_rddl_path,
        graph_index_path=graph_index_path,
        n_simulations=args.n_simulations, 
        log_window=args.log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        rddl_path=args.rddl_path, 
        tmp_path=args.tmp_path,
        snapshot_sequence_path=args.snapshot_sequence_path,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    s = f'Training data produced and written to {training_sequence_path}.'
    logging.info(s)
    print(s)

if 'eval_seq' in args.modes:
    if instance_rddl_path is None:
        instance_rddl_path = args.instance_rddl_path
    if graph_index_path is None:
        graph_index_path = args.graph_index_path
    else:
        logging.info(f'Using newly generated instance {instance_rddl_path} and graph index {graph_index_path}.')
    logging.info(f'Producing single evaluation snapshot sequence.')
    simulator = Simulator()
    evaluation_sequence_path = simulator.produce_training_data_parallel(
        domain_rddl_path=args.domain_rddl_path,
        instance_rddl_path=instance_rddl_path,
        graph_index_path=graph_index_path,
        n_simulations=1, 
        log_window=args.log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        rddl_path=args.rddl_path, 
        tmp_path=args.tmp_path,
        snapshot_sequence_path=args.snapshot_sequence_path,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    s = f'Training data produced and written to {training_sequence_path}.'
    logging.info(s)
    print(s)

if 'train' in args.modes:
    if training_sequence_path is None:
        training_sequence_path = args.training_sequence_file_name
    else:
        logging.info(f'Using newly generated training sequence {training_sequence_path}.')
    logging.info(f'Training GNN.')
    predictor_filename = train_gnn(
                    number_of_epochs=args.epochs, 
                    sequence_file_name=args.training_sequence_file_name, 
                    learning_rate=args.learning_rate, 
                    batch_size=args.batch_size, 
                    hidden_layers_list=hidden_layers)
    s = f'GNN trained. Model written to {predictor_filename}.'
    logging.info(s)
    print(s)

if 'evaluate' in args.modes:
    if evaluation_sequence_path is None:
        evaluation_sequence_path = args.evaluation_sequence_path
    if predictor_filename is None:
        predictor_filename = args.predictor_filename
    else:
        logging.info(f'Using newly generated predictor {predictor_filename}.')
    evaluator = Evaluator(trigger_threshold=0.5)
    evaluator.evaluate_test_set(
        args.predictor_type, 
        predictor_filename, 
        args.evaluation_sequence_path)

if 'animate' in args.modes:
    logging.info(f'Creating animation.')
    if evaluation_sequence_path is None:
        evaluation_sequence_path = args.evaluation_sequence_path
        animation_sequence_filename = args.animation_sequence_filename
    else:
        animation_sequence_filename = evaluation_sequence_path
    if predictor_filename is None:
        predictor_filename = args.predictor_filename
    else:
        logging.info(f'Using newly generated predictor {predictor_filename}.')
    animator = Animator(animation_sequence_filename)
    animator.create_animation(predictor_type=args.predictor_type, 
                             predictor_filename=predictor_filename)
    s = f'Animation written to file network_animation.gif.'
    logging.info(s)
    print(s)


