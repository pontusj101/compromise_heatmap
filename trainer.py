import time
import numpy
import random
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
from pyRDDLGym.Core.Policies.Agents import RandomAgent
from agents import PassiveCyberAgent
from agents import RandomCyberAgent
import torch
from torch_geometric.data import Data

node_mapping = {
    'observed_crack_attack___c1': 6,
    'observed_crack_attack___c2': 7,
    'observed_crack_attack___c3': 8,
    'observed_crack_attack___c4': 9,
    'observed_crack_attack___c5': 10,
    'observed_crack_attack___c6': 11}


# Node features
node_features = torch.tensor([
    [1], [1], [1], [1], [1], [1],  # Host features
    [0], [0], [0], [0], [0], [0]   # Credential features
])

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

N = len(node_features)
time_steps = 25  # Number of time steps

data_series = []


base_path = 'content/'
myEnv = RDDLEnv.RDDLEnv(domain=base_path+'domain.rddl', instance=base_path+'instance.rddl')

start_step =  random.randint(time_steps, 100)
print(f'Starting attack at step {start_step}')
agent = PassiveCyberAgent(action_space=myEnv.action_space)
# agent = RandomCyberAgent(action_space=myEnv.action_space, seed=42)


log_trace = []
total_reward = 0
state = myEnv.reset()
start_time = time.time()
print(f'step         = 0')
print(f'attack steps = {[attackstep for attackstep, value in state.items() if type(value) is numpy.bool_ and value == True]}')
print(f'TTCs         = {[(attackstep, value) for attackstep, value in state.items() if type(value) is numpy.int64]}')
for step in range(myEnv.horizon):
    if step == start_step:
        agent = RandomCyberAgent(action_space=myEnv.action_space, seed=42)
        print(f'Now initiating attack.')
    action = agent.sample_action()
    next_state, reward, done, info = myEnv.step(action)
    observations = [key for key, value in next_state.items() if type(value) is numpy.bool_ and value == True and "observed" in key]
    log_trace.append(observations)
    truncated_log_trace = log_trace[-time_steps:]
    attacksteps = [key for key, value in next_state.items() if type(value) is numpy.bool_ and value == True and "observed" not in key]
    total_reward += reward
    print()
    print(f'step              = {step}')
    print(f'action            = {action}')
    print(f'observations      = {observations}')
    print(f'log trace (trunc) = {truncated_log_trace}')
 #   print(f'log trace         = {log_trace}')
    print(f'attack steps      = {attacksteps}')
    print(f'TTCs              = {[(attackstep, value) for attackstep, value in next_state.items() if type(value) is numpy.int64]}')
    print(f'reward            = {reward}')
    state = next_state

    if step >= time_steps:
        print(f'Now initiating logging (having populated the log with non-malicious data).')

        # Initialize feature vectors with -1
        log_feature_vectors = torch.full((N, time_steps), -1)

        # Update feature vectors based on log traces
        for t, log_t in enumerate(truncated_log_trace):
            for node in range(N):
                # Check if node is observable at this time step
                if any(log_event in log_t for log_event, idx in node_mapping.items() if idx == node):
                    log_feature_vectors[node][t] = 1
                elif log_feature_vectors[node][t] == -1:
                    log_feature_vectors[node][t] = 0

        combined_features = torch.cat((node_features, log_feature_vectors), dim=1)

        # Create PyTorch Geometric data object
        data = Data(x=combined_features, edge_index=edge_index)

        # Print the conttent of the data object
        print(data.x)
        print(data.edge_index)

        data_series.append(data)

    if done:
        break
end_time = time.time()
print()
print(f'episode ended with reward {total_reward}. Execution time was {end_time-start_time} s.')

myEnv.close()


