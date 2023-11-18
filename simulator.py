import time
import random
import pickle
import numpy
import torch
from torch_geometric.data import Data
from pyRDDLGym import RDDLEnv
from agents import PassiveCyberAgent, RandomCyberAgent

def torch_data_from_state(state, log_mapping, attackstep_mapping, log_window, debug_print=False):
    # Node features
    node_features = torch.tensor([
        [1], [1], [1], [1], [1], [1],  # Host features
        [0], [0], [0], [0], [0], [0]   # Credential features
    ], dtype=torch.float)

    # Edges
    edge_index = torch.tensor([
        # CONNECTED Edges (Host-to-Host) and ACCESSES/STORES Edges (Credentials-to-Host, Host-to-Credentials)
        [0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 0, 5, 1, 3, 3, 4],
        [1, 2, 3, 5, 4, 5, 0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 8, 11]
    ], dtype=torch.long)

    n_nodes = len(node_features)
    log_feature_vectors = torch.zeros((n_nodes, log_window))
    labels = torch.zeros(n_nodes)

    for key, value in state.items():
        if isinstance(value, numpy.bool_):
            if "observed" in key:
                node_index = log_mapping.get(key)
                if node_index is not None:
                    log_feature_vectors[node_index] = torch.roll(log_feature_vectors[node_index], -1)
                    log_feature_vectors[node_index][-1] = 1.0
            else:
                node_index = attackstep_mapping.get(key)
                if node_index is not None:
                    labels[node_index] = 1.0

    # Convert labels to torch.long
    labels = labels.to(torch.long)

    combined_features = torch.cat((node_features, log_feature_vectors), dim=1)
    data = Data(x=combined_features, edge_index=edge_index, y=labels)

    if debug_print:
        print(data)

    return data

def produce_training_data(use_saved_data=True, n_simulations=10, log_window=25, max_start_time_step=100, rddl_path='content/', random_cyber_agent_seed=42, debug_print=False):
    file_name = 'data_series.pkl'
    if use_saved_data:
        with open(file_name, 'rb') as file:
            data_series = pickle.load(file)
    else:
        data_series = []
        myEnv = RDDLEnv.RDDLEnv(domain=rddl_path+'domain.rddl', instance=rddl_path+'instance.rddl', render_mode='none')

        log_mapping = {
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

        attackstep_mapping = {
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

        for i_sims in range(n_simulations):
            start_step = random.randint(log_window, max_start_time_step)
            agent = PassiveCyberAgent(action_space=myEnv.action_space)

            state = myEnv.reset()
            start_time = time.time()
            if debug_print:
                print(f'Starting attack at step {start_step}')

            for step in range(myEnv.horizon):
                if step == start_step:
                    agent = RandomCyberAgent(action_space=myEnv.action_space, seed=random_cyber_agent_seed)
                    if debug_print:
                        print('Now initiating attack.')

                action = agent.sample_action()
                next_state, reward, done, info = myEnv.step(action)

                if step >= log_window:
                    data = torch_data_from_state(next_state, log_mapping, attackstep_mapping, log_window, debug_print=debug_print)
                    data_series.append(data)

                if done:
                    break

            end_time = time.time()
            print(f'Simulation {i_sims}: episode ended with reward {reward}. Execution time was {end_time - start_time} s.')

        myEnv.close()

        with open(file_name, 'wb') as file:
            pickle.dump(data_series, file)
        print(f'Data saved to {file_name}')

    return data_series


