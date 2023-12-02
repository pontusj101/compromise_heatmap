import argparse
import logging
from animator import Animator
from instance_creator import create_instance
from simulator import Simulator
from evaluator import Evaluator
from gnn_trainer import train_gnn

# Initialize parser
parser = argparse.ArgumentParser(description='Run different modes of the security simulation program.')

# Adding arguments
parser.add_argument('mode', choices=['instance', 'simulate', 'train', 'animate', 'evaluate'], help='Mode of operation')
parser.add_argument('--instance_type', default='static', choices=['static', 'dynamic'], help='Type of instance to create')
parser.add_argument('--size', default='small', choices=['small', 'medium', 'large'], help='Size of the graph')
parser.add_argument('--game_time', type=int, default=70, help='Time horizon for the simulation')
parser.add_argument('-n', '--n_simulations', type=int, default=16, help='Number of simulations to run')
parser.add_argument('-l', '--log_window', type=int, default=3, help='Size of the logging window')
parser.add_argument('--random_cyber_agent_seed', default=None, help='Seed for random cyber agent')
parser.add_argument('--number_of_epochs', type=int, default=8, help='Number of epochs for GNN training')
parser.add_argument('--learning_rate', type=float, default=0.005, help='Learning rate for GNN training')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size for GNN training')
parser.add_argument('--hidden_layers', nargs='+', type=int, default=[64, 64], help='Hidden layers configuration for GNN')
parser.add_argument('--rddl_path', default='rddl/', help='Path to the RDDL files')
parser.add_argument('--tmp_path', default='tmp/', help='Temporary file path')
parser.add_argument('--domain_rddl_path', default='rddl/domain.rddl', help='Path to RDDL domain specification')
parser.add_argument('--instance_rddl_path', default='rddl/instance_static_small.rddl' , help='Path to RDDL instance specification')
parser.add_argument('--graph_index_path', default='rddl/graph_index_static_small.pkl', help='Path to pickled GraphIndex class.')
parser.add_argument('--snapshot_sequence_path', default='snapshot_sequences/', help='Path to snapshot sequences')
parser.add_argument('--training_sequence_file_name', default='snapshot_sequences/snapshot_sequence_1024_static_small.pkl', help='Filename for training sequence')
parser.add_argument('--evaluation_sequence_file_name', default='snapshot_sequences/snapshot_sequence_1_static_small.pkl', help='Filename for evaulation sequence')
parser.add_argument('--trigger_threashold', type=float, default=0.5, help='The threashold probability at which a predicted label is considered positive.')
parser.add_argument('--animation_sequence_filename', default='snapshot_sequences/snapshot_sequence_1_static_small.pkl', help='Filename for animation sequence')
parser.add_argument('--predictor_filename', default='models/model_hl_[64, 64]_n_50390_lr_0.005_bs_256.pt', help='Filename for the predictor model')
parser.add_argument('--predictor_type', default='gnn', choices=['gnn', 'tabular', 'none'], help='Type of predictor')

# Parse arguments
args = parser.parse_args()

# Set up logging
logging.basicConfig(filename='log.log', level=logging.DEBUG, format='%(asctime)s - %(message)s')
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.warning('\n\n')

max_start_time_step = args.log_window + int((args.game_time - args.log_window) / 2)
max_log_steps_after_total_compromise = int(args.log_window / 2)

# Modes
if args.mode == 'create_instance':
    # TODO: Write graph_index to file 
    logging.info(f'Creating new instance specification.')
    rddl_file_path, graph_index_file_path = create_instance(
        instance_type=args.instance_type, 
        size=args.size, 
        horizon=args.game_time, 
        rddl_path=args.rddl_path)
    s = f'Instance specification written to {rddl_file_path}. Graph index written to {graph_index_file_path}.'
    logging.info(s)
    print(s)

elif args.mode == 'produce_training_data':
    logging.info(f'Producing training data.')
    simulator = Simulator()
    sequence_file_name = simulator.produce_training_data_parallel(
        domain_rddl_path=args.domain_rddl_path,
        instance_rddl_path=args.instance_rddl_path,
        graph_index_path=args.graph_index_path,
        n_simulations=args.n_simulations, 
        log_window=args.log_window, 
        max_start_time_step=max_start_time_step, 
        max_log_steps_after_total_compromise=max_log_steps_after_total_compromise,
        rddl_path=args.rddl_path, 
        tmp_path=args.tmp_path,
        snapshot_sequence_path=args.snapshot_sequence_path,
        random_cyber_agent_seed=args.random_cyber_agent_seed)
    s = f'Training data produced and written to {sequence_file_name}.'
    logging.info(s)
    print(s)

elif args.mode == 'train_gnn':
    logging.info(f'Training GNN.')
    predictor_filename = train_gnn(
                    number_of_epochs=args.number_of_epochs, 
                    sequence_file_name=args.training_sequence_file_name, 
                    learning_rate=args.learning_rate, 
                    batch_size=args.batch_size, 
                    hidden_layers=args.hidden_layers)
    s = f'GNN trained. Model written to {predictor_filename}.'
    logging.info(s)
    print(s)

elif args.mode == 'animate':
    logging.info(f'Creating animation.')
    animator = Animator(args.animation_sequence_filename)
    animator.create_animation(predictor_type=args.predictor_type, 
                             predictor_filename=args.predictor_filename)
    s = f'Animation written to file network_animation.gif.'
    logging.info(s)
    print(s)

elif args.mode == 'evaluate':
    logging.info(f'Evaluating {args.predictor_type} predictor {args.predictor_filename} on {args.evaluation_sequence_file_name}.')
    evaluator = Evaluator(trigger_threshold=0.5)
    evaluator.evaluate_test_set(args.predictor_type, args.predictor_filename, args.evaluation_sequence_file_name)