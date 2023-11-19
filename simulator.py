import multiprocessing
import time
import random
import pickle
import numpy
import torch
from torch_geometric.data import Data
from pyRDDLGym import RDDLEnv
from agents import PassiveCyberAgent, RandomCyberAgent
from instance_creator import create_instance

class GraphIndex:
    def __init__(self, size='small'):
        if size == 'small':
            self.log_mapping = {
                'observed_compromise_attack___h1': 0,
                'observed_compromise_attack___h2': 1}

            self.attackstep_mapping = {
                'compromised___h1': 0,
                'compromised___h2': 1}

            # Node features
            self.node_features = torch.tensor([
                [1], [1],  # Host features
                [0], [0]   # Credential features
            ], dtype=torch.float)

            # Edges
            self.edge_index = torch.tensor([
                # CONNECTED Edges (Host-to-Host) and ACCESSES/STORES Edges (Credentials-to-Host, Host-to-Credentials)
                [0, 0, 0, 2, 3],
                [1, 2, 3, 0, 1]
            ], dtype=torch.long)
        elif size == 'medium':
            self.log_mapping = {
                'observed_compromise_attack___h1': 0,
                'observed_compromise_attack___h2': 1,
                'observed_compromise_attack___h3': 2,
                'observed_compromise_attack___h4': 3,
                'observed_compromise_attack___h5': 4,
                'observed_compromise_attack___h6': 5,
                'observed_crack_attack___c1': 6,
                'observed_crack_attack___c2': 7,
                'observed_crack_attack___c3': 8,
                'observed_crack_attack___c4': 9,
                'observed_crack_attack___c5': 10,
                'observed_crack_attack___c6': 11}

            self.attackstep_mapping = {
                'compromised___h1': 0,
                'compromised___h2': 1,
                'compromised___h3': 2,
                'compromised___h4': 3,
                'compromised___h5': 4,
                'compromised___h6': 5,
                'cracked___c1': 6,
                'cracked___c2': 7,
                'cracked___c3': 8,
                'cracked___c4': 9,
                'cracked___c5': 10,
                'cracked___c6': 11}

            self.node_features = torch.tensor([
                [1], [1], [1], [1], [1], [1],  # Host features
                [0], [0], [0], [0], [0], [0]   # Credential features
            ], dtype=torch.float)

            # Edges
            self.edge_index = torch.tensor([
                # CONNECTED Edges (Host-to-Host) and ACCESSES/STORES Edges (Credentials-to-Host, Host-to-Credentials)
                [0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 5, 1, 3, 3, 4],
                [1, 2, 3, 5, 4, 5, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 11]
            ], dtype=torch.long)
        elif size == 'large':
            print('Not implemented yet')

def simulation_worker(sim_id, log_window, max_start_time_step, graph_size, rddl_path, random_cyber_agent_seed, debug_print):
    myEnv = RDDLEnv.RDDLEnv(domain=rddl_path+'domain.rddl', instance=rddl_path+'instance.rddl')

    graph_index = GraphIndex(size=graph_size)

    n_nodes = len(graph_index.node_features)

    start_step = random.randint(log_window, max_start_time_step)
    agent = PassiveCyberAgent(action_space=myEnv.action_space)

    state = myEnv.reset()
    total_reward = 0
    start_time = time.time()
    if debug_print >= 2:
        print(f'Starting attack at step {start_step} in simulation {sim_id}')

    snapshot_sequence = []
    log_feature_vectors = torch.zeros((n_nodes, log_window))
    for step in range(myEnv.horizon):
        if step == start_step:
            agent = RandomCyberAgent(action_space=myEnv.action_space, seed=random_cyber_agent_seed)
            if debug_print >= 2:
                print('Now initiating attack.')

        action = agent.sample_action()
        state, reward, done, info = myEnv.step(action)
        total_reward += reward
        if step >= log_window:

            labels = torch.zeros(n_nodes)
            log_line = torch.zeros(n_nodes)

            for key, value in state.items():
                if isinstance(value, numpy.bool_):
                    if "observed" in key:
                        node_index = graph_index.log_mapping.get(key)
                        if node_index is not None:
                            if value:
                                log_line[node_index] = 1.0
                            else:
                                log_line[node_index] = 0.0
                    else:
                        node_index = graph_index.attackstep_mapping.get(key)
                        if node_index is not None:
                            if value:
                                labels[node_index] = 1.0
                            else:
                                labels[node_index] = 0.0

            log_feature_vectors = torch.cat((log_feature_vectors[:, 1:], log_line.unsqueeze(1)), dim=1)
            if debug_print >= 2:
                print(f'The compromised steps are {labels}')
                print(f'The most recent log line is {log_line}')
                print(f'The complete log is \n{log_feature_vectors}')



            # Convert labels to torch.long
            labels = labels.to(torch.long)

            combined_features = torch.cat((graph_index.node_features, log_feature_vectors), dim=1)
            snapshot = Data(x=combined_features, edge_index=graph_index.edge_index, y=labels)

            if debug_print >= 2:
                print(f'At step {step} the snapshot is:')
                print(snapshot.x)
                print(snapshot.edge_index)
                print(snapshot.y)

            snapshot_sequence.append(snapshot)

        if done:
            break

    end_time = time.time()
    if debug_print >= 2:
        print(f'Simulation {sim_id}: episode ended with reward {total_reward}. Execution time was {end_time - start_time} s.')
        print('20th snapshot (x [node features + log data], edge_index and y [attack steps]):')
        print(snapshot_sequence[20].x)
        print(snapshot_sequence[20].edge_index)
        print(snapshot_sequence[20].y)

    myEnv.close()

    return snapshot_sequence


def produce_training_data_parallel(use_saved_data=False, 
                                   n_simulations=10, 
                                   log_window=25, 
                                   game_time=200, 
                                   max_start_time_step=100, 
                                   graph_size='small', 
                                   rddl_path='content/', 
                                   random_cyber_agent_seed=42, 
                                   debug_print=1):
    file_name = 'data_series_parallel.pkl'
    if use_saved_data:
        with open(file_name, 'rb') as file:
            data_series = pickle.load(file)
    else:
        create_instance(size=graph_size, horizon=game_time, rddl_path=rddl_path)
        n_processes = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=n_processes)

        simulation_args = [(i, log_window, max_start_time_step, graph_size, rddl_path, random_cyber_agent_seed, debug_print) for i in range(n_simulations)]

        results = pool.starmap(simulation_worker, simulation_args)
        pool.close()
        pool.join()

        # Flatten the list of lists (each sublist is the output from one simulation)
        data_series = [item for sublist in results for item in sublist]

        with open(file_name, 'wb') as file:
            pickle.dump(data_series, file)
        print(f'Data saved to {file_name}')

    return data_series
