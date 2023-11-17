import time
import random
import numpy
import pickle
from pyRDDLGym import RDDLEnv
import torch
from torch_geometric.data import Data
from agents import PassiveCyberAgent
from agents import RandomCyberAgent

def torch_data_from_trace(truncated_log_trace, attacksteps, debug_print=False):
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

    # Node features
    node_features = torch.tensor([
        [1], [1], [1], [1], [1], [1],  # Host features
        [0], [0], [0], [0], [0], [0]   # Credential features
    ], dtype=torch.float)

    # Edges
    edge_index = torch.tensor([
        # CONNECTED Edges (Host-to-Host)
        [0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 5, 
        # ACCESSES Edges (Credentials-to-Host)
        6, 7, 8, 9, 10, 11, 
        # STORES Edges (Host-to-Credentials)
        0, 0, 5, 1, 3, 3, 4],
        [1, 2, 3, 5, 4, 5, 0, 0, 1, 2, 3, 
        0, 1, 2, 3, 4, 5, 
        6, 7, 8, 9, 10, 8, 11]
    ], dtype=torch.long)

    n_nodes = len(node_features)
    time_steps = len(truncated_log_trace)
    # Initialize feature vectors with -1
    log_feature_vectors = torch.full((n_nodes, time_steps), -1)

    # Initialize labels with 0 (uncompromised)
    labels = torch.zeros(n_nodes)

    # Update feature vectors based on log traces
    for t, log_t in enumerate(truncated_log_trace):
        for node in range(n_nodes):
            # Check if node is observable at this time step
            if any(log_event in log_t for log_event, idx in log_mapping.items() if idx == node):
                log_feature_vectors[node][t] = 1
            elif log_feature_vectors[node][t] == -1:
                log_feature_vectors[node][t] = 0

    # Update labels based on current state
    for attackstep in attacksteps:
        if attackstep in attackstep_mapping:
            labels[attackstep_mapping[attackstep]] = 1

    labels = labels.to(torch.long)
    log_feature_vectors = log_feature_vectors.to(torch.float)
    combined_features = torch.cat((node_features, log_feature_vectors), dim=1)

    # Create PyTorch Geometric data object
    data = Data(x=combined_features, edge_index=edge_index, y=labels)

    # Print the conttent of the data object
    if debug_print:
        print(data.x)
        print(data.edge_index)
        print(data.y)

    return data

def produce_training_data(use_saved_data=True, n_simulations=10, log_window=25, max_start_time_step=100, rddl_path='content/', random_cyber_agent_seed=42, debug_print=False):
    file_name = 'data_series.pkl'
    if use_saved_data:
        with open(file_name, 'rb') as file:
            data_series = pickle.load(file)
    else:

        data_series = []

        myEnv = RDDLEnv.RDDLEnv(domain=rddl_path+'domain.rddl', instance=rddl_path+'instance.rddl')

        for i_sims in range(n_simulations):
            start_step =  random.randint(log_window, max_start_time_step)
            agent = PassiveCyberAgent(action_space=myEnv.action_space)

            log_trace = []
            total_reward = 0
            state = myEnv.reset()
            start_time = time.time()
            if debug_print:
                print(f'Starting attack at step {start_step}')
                print(f'step         = 0')
                print(f'attack steps = {[attackstep for attackstep, value in state.items() if type(value) is numpy.bool_ and value == True]}')
                print(f'TTCs         = {[(attackstep, value) for attackstep, value in state.items() if type(value) is numpy.int64]}')
            for step in range(myEnv.horizon):
                if step == start_step:
                    agent = RandomCyberAgent(action_space=myEnv.action_space, seed=random_cyber_agent_seed)
                    if debug_print:
                        print(f'Now initiating attack.')
                action = agent.sample_action()
                next_state, reward, done, info = myEnv.step(action)
                observations = [key for key, value in next_state.items() if type(value) is numpy.bool_ and value == True and "observed" in key]
                log_trace.append(observations)
                truncated_log_trace = log_trace[-log_window:]
                attacksteps = [key for key, value in next_state.items() if type(value) is numpy.bool_ and value == True and "observed" not in key]
                total_reward += reward
                state = next_state
                if debug_print:
                    print()
                    print(f'step              = {step}')
                    print(f'action            = {action}')
                    print(f'observations      = {observations}')
                    print(f'log trace (trunc) = {truncated_log_trace}')
                    print(f'attack steps      = {attacksteps}')
                    print(f'TTCs              = {[(attackstep, value) for attackstep, value in next_state.items() if type(value) is numpy.int64]}')
                    print(f'reward            = {reward}')

                if step >= log_window:
                    if debug_print:
                        print(f'Now initiating logging (having populated the beginning of the log with non-malicious data).')

                    data = torch_data_from_trace(truncated_log_trace, attacksteps, debug_print=debug_print)
                    data_series.append(data)

                if done:
                    break

            end_time = time.time()
            print(f'Simulation {i_sims}: episode ended with reward {total_reward}. Execution time was {end_time-start_time} s.')

        myEnv.close()

        # Choose a file name
        with open(file_name, 'wb') as file:
            pickle.dump(data_series, file)
        print(f'Data saved to {file_name}')

    return data_series

