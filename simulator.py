import multiprocessing
import os
import re
import time
from datetime import datetime
import random
import numpy
import torch
import logging
import numpy as np
from torch_geometric.data import Data
from pyRDDLGym import RDDLEnv
from agents import PassiveCyberAgent, RandomCyberAgent
from graph_index import GraphIndex

class Simulator:

    def vectorized_labels(self, state, graph_index):
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

    def vectorized_log_line(self, state, graph_index):
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

    def simulation_worker(self, 
                          sim_id, 
                          log_window, 
                          max_start_time_step, 
                          max_log_steps_after_total_compromise, 
                          graph_index_filepath, 
                          domain_rddl_path, 
                          instance_rddl_filepath, 
                          tmp_path, 
                          random_cyber_agent_seed):
        
        myEnv = RDDLEnv.RDDLEnv(domain=domain_rddl_path, instance=instance_rddl_filepath)
        graph_index = torch.load(graph_index_filepath)

        start_time = time.time()
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
                logging.debug(f'Simulation {sim_id}. Step {step}: Now initiating attack.')

            action = agent.sample_action(state=state)
            state, reward, done, info = myEnv.step(action)
            total_reward += reward

            log_line = self.vectorized_log_line(state, graph_index)
            log_feature_vectors = torch.cat((log_feature_vectors[:, 1:], log_line.unsqueeze(1)), dim=1)
            labels = self.vectorized_labels(state, graph_index)
            labels = labels.to(torch.long)
            if (labels == 1).all():
                if log_steps_after_total_compromise == 0:
                    logging.debug(f'Simulation {sim_id}. Step {step}: All attack steps were compromised. Continuing to log for {max_log_steps_after_total_compromise} steps.')
                log_steps_after_total_compromise += 1
                if log_steps_after_total_compromise > max_log_steps_after_total_compromise:
                    logging.debug(f'Simulation {sim_id}. Step {step}: Logged {max_log_steps_after_total_compromise} snapshots after all steps were compromised. Terminating simulation.')
                    break
            combined_features = torch.cat((graph_index.node_features, log_feature_vectors), dim=1)
            snapshot = Data(x=combined_features, edge_index=graph_index.edge_index, edge_type=graph_index.edge_type, y=labels)

            # Only add snapshots after the log window has been filled with unmalicious log lines
            if step >= log_window:
                snapshot_sequence.append(snapshot)

            if done:
                break

        myEnv.close()
        end_time = time.time()
        indexed_snapshot_sequence = {'snapshot_sequence': snapshot_sequence, 'graph_index': graph_index}

        output_file = os.path.join(tmp_path, f"simulation_{sim_id}.pkl")
        torch.save(indexed_snapshot_sequence, output_file)
        
        return output_file



    def produce_training_data_parallel(
        self, 
        domain_rddl_path,
        instance_rddl_filepaths,
        graph_index_filepaths,
        log_window=25, 
        max_start_time_step=100, 
        max_log_steps_after_total_compromise=50,
        rddl_path='rddl/', 
        tmp_path='tmp/',
        snapshot_sequence_path = 'snapshot_sequences/',
        random_cyber_agent_seed=None):
        
        start_time = time.time()


        n_simulations = len(instance_rddl_filepaths)
        n_processes = multiprocessing.cpu_count()
        result_filenames = []
        logging.info(f'Starting simulation of {n_simulations} instance models and a log window of {log_window}.')
        pool = multiprocessing.Pool(processes=n_processes)

        simulation_args = [(i, log_window, max_start_time_step, max_log_steps_after_total_compromise, graph_index_filepaths[i], domain_rddl_path, instance_rddl_filepaths[i], tmp_path, random_cyber_agent_seed) for i in range(n_simulations)]

        result_filenames = pool.starmap(self.simulation_worker, simulation_args)
        pool.close()
        pool.join()

        results = []
        for output_file in result_filenames:
            results.append(torch.load(output_file))
            os.remove(output_file) 

        n_completely_compromised = sum([(snap_seq['snapshot_sequence'][-1].y[1:] == 1).all() for snap_seq in results])


        match = re.search(r'instance_(.*?)\.rddl', instance_rddl_filepaths[0])
        instance_name = match.group(1) if match else None


        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f'{snapshot_sequence_path}snapshot_sequence_n{n_simulations}_l{log_window}_{instance_name}.pkl'
        torch.save(results, file_name)
        logging.info(f'Data saved to {file_name}')

        snapshot_sequence = [item for sublist in [r['snapshot_sequence'] for r in results] for item in sublist]
        compromised_snapshots = sum(tensor.sum() > 1 for tensor in [s.y for s in snapshot_sequence])
        logging.info(f'Training data generation completed. Time: {time.time() - start_time:.2f}s.')
        logging.info(f'Number of snapshots: {len(snapshot_sequence)}, of which {compromised_snapshots} are compromised, and {n_completely_compromised} of {n_simulations} simulations ended in complete compromise.')
        random_snapshot_index = np.random.randint(0, len(snapshot_sequence))
        random_snapshot = snapshot_sequence[random_snapshot_index]
        logging.debug(f'Random snapshot ({random_snapshot_index}) node features, log sequence and labels:')
        logging.debug(f'\n{random_snapshot.x[:,:1]}')
        logging.debug(f'\n{random_snapshot.x[:,1:]}')
        logging.debug(random_snapshot.y)


        return file_name
