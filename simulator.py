import multiprocessing
import time
import random
import pickle
import numpy
import torch
import logging
from torch_geometric.data import Data
from pyRDDLGym import RDDLEnv
from agents import PassiveCyberAgent, RandomCyberAgent
from instance_creator import create_instance
from graph_index import GraphIndex


def vectorized_labels(state, graph_index):
    n_attacksteps = len(graph_index.attackstep_mapping)
    labels = torch.zeros(n_attacksteps)
    n_detectors = len(graph_index.log_mapping)
    log_line = torch.zeros(n_detectors)
    for key, value in state.items():
        if isinstance(value, numpy.bool_):
            if not "observed" in key:
                node_index = graph_index.attackstep_mapping.get(key)
                if node_index is not None:
                    if value:
                        labels[node_index] = 1.0
                    else:
                        labels[node_index] = 0.0
    return labels

def vectorized_log_line(state, graph_index):
    n_detectors = len(graph_index.log_mapping)
    log_line = torch.zeros(n_detectors)
    for key, value in state.items():
        if isinstance(value, numpy.bool_):
            if "observed" in key:
                node_index = graph_index.log_mapping.get(key)
                if node_index is not None:
                    if value:
                        log_line[node_index] = 1.0
                    else:
                        log_line[node_index] = 0.0
    return log_line

def simulation_worker(sim_id, log_window, max_start_time_step, graph_size, rddl_path, random_cyber_agent_seed):
    myEnv = RDDLEnv.RDDLEnv(domain=rddl_path+'domain.rddl', instance=rddl_path+'instance.rddl')

    start_time = time.time()
    graph_index = GraphIndex(size=graph_size)
    n_nodes = len(graph_index.node_features)
    start_step = random.randint(log_window, max_start_time_step)

    agent = PassiveCyberAgent(action_space=myEnv.action_space)
    state = myEnv.reset()
    total_reward = 0
    snapshot_sequence = []
    log_feature_vectors = torch.zeros((n_nodes, log_window))
    log_steps_after_total_compromise = 0
    for step in range(myEnv.horizon):
        if step == start_step:
            agent = RandomCyberAgent(action_space=myEnv.action_space, seed=random_cyber_agent_seed)
            logging.info(f'Step {step}: Now initiating attack.')

        action = agent.sample_action()
        state, reward, done, info = myEnv.step(action)
        total_reward += reward

        log_line = vectorized_log_line(state, graph_index)
        log_feature_vectors = torch.cat((log_feature_vectors[:, 1:], log_line.unsqueeze(1)), dim=1)
        labels = vectorized_labels(state, graph_index)
        labels = labels.to(torch.long)
        if (labels == 1).all():
            if log_steps_after_total_compromise == 0:
                logging.info(f'Step {step}: All attack steps were compromised. Continuing to log for {log_window/2} steps.')
            log_steps_after_total_compromise += 1
            if log_steps_after_total_compromise > log_window/2:
                logging.info(f'Step {step}: All attack steps were compromised. Terminating simulation.')
                break
        combined_features = torch.cat((graph_index.node_features, log_feature_vectors), dim=1)
        snapshot = Data(x=combined_features, edge_index=graph_index.edge_index, y=labels)

        # Only add snapshots after the log window has been filled with unmalicious log lines
        if step >= log_window:
            snapshot_sequence.append(snapshot)

        if done:
            break

    myEnv.close()
    end_time = time.time()

    return snapshot_sequence


def produce_training_data_parallel(use_saved_data=False, 
                                   n_simulations=10, 
                                   log_window=25, 
                                   game_time=200, 
                                   max_start_time_step=100, 
                                   graph_size='small', 
                                   rddl_path='content/', 
                                   random_cyber_agent_seed=None):
    file_name = 'data_series_parallel.pkl'
    if use_saved_data:
        with open(file_name, 'rb') as file:
            data_series = pickle.load(file)
            logging.info(f'Data retrieved from file {file_name}')
    else:
        create_instance(size=graph_size, horizon=game_time, rddl_path=rddl_path)
        n_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_processes)

        simulation_args = [(i, log_window, max_start_time_step, graph_size, rddl_path, random_cyber_agent_seed) for i in range(n_simulations)]

        results = pool.starmap(simulation_worker, simulation_args)
        pool.close()
        pool.join()

        # Flatten the list of lists (each sublist is the output from one simulation)
        data_series = [item for sublist in results for item in sublist]

        with open(file_name, 'wb') as file:
            pickle.dump(data_series, file)
        logging.info(f'Data saved to {file_name}')

    return data_series
