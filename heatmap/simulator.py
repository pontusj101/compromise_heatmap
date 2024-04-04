import multiprocessing
import os
import io
import re
import time
import logging
from datetime import datetime
import random
import numpy
import torch
import numpy as np
# from memory_profiler import profile
from google.cloud import storage
from google.cloud.storage.retry import DEFAULT_RETRY
from torch_geometric.data import Data
from malpzsim.agents.searchers import BreadthFirstAttacker, PassiveAttacker
from maltoolbox.language import specification, LanguageGraph, LanguageClassesFactory
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model
from malpzsim.sims.mal_petting_zoo_simulator import MalPettingZooSimulator
from malpzsim.agents.searchers import BreadthFirstAttacker

from . import logging as heatmap_logging

logger = logging.getLogger(__name__)

class Simulator:
    stdin = None

    def simulation_worker(self,
                          sim_id,
                          bucket_name,
                          log_window,
                          max_start_time_step,
                          max_log_steps_after_total_compromise,
                          domain_rddl_path,
                          instance_rddl_filepath,
                          storage_path,
                          cyber_agent_type='random',
                          novelty_priority=2,
                          random_cyber_agent_seed=None):
        import sys
        sys.stdin = open("/dev/tty")
        heatmap_logging.debug = '--debug' in sys.argv
        logger = heatmap_logging.setup_logging()

        logger.info(f'Simulation {sim_id} started.')
#
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)
        # instance_blob = bucket.blob(instance_rddl_filepath)
        # local_instance_filepath = f'/tmp/instance_{sim_id}.rddl'
        # instance_blob.download_to_filename(local_instance_filepath)
        # instance_blob.delete()
        # gi_blob.delete()

        lang_graph = LanguageGraph.from_mar_archive(domain_rddl_path)
        lang_classes_factory = LanguageClassesFactory(lang_graph)
        lang_classes_factory.create_classes()

        model = Model.load_from_file("../mal-petting-zoo-simulator/tests/example_model.json", lang_classes_factory)
        model.save_to_file("tmp/model.json")

        attack_graph = AttackGraph(lang_graph, model)
        attack_graph.attach_attackers()
        attack_graph.save_to_file("tmp/attack_graph.json")

        # max_iter is the episode horizon
        horizon = 400
        myEnv = MalPettingZooSimulator(lang_graph, model, attack_graph, max_iter=horizon)

        start_time = time.time()
        n_nodes = len(attack_graph.nodes)
        # start_step = random.randint(log_window, max_start_time_step)
        start_step = random.randint(log_window, 100)

        logger.info(f"start step: {start_step}")

        agent = PassiveAttacker()
        total_reward = 0
        snapshot_sequence = []
        log_feature_vectors = torch.zeros((n_nodes, log_window))
        log_steps_after_total_compromise = myEnv.horizon = horizon

        myEnv.register_attacker("attacker", 0)
        myEnv.register_defender("defender")

        state, infos = myEnv.reset()
        labels = None
        log_line = None


        for step in range(myEnv.horizon):
            if step % 30 == 0:
                logger.info(f'Simulation {sim_id}. Step {step}/{myEnv.horizon}. Time: {time.time() - start_time:.2f}s.')
            if step == start_step:
                if cyber_agent_type == 'random':
                    agent = BreadthFirstAttacker({"randomize": True})
                    logger.info(f'Simulation {sim_id}. Deploying random attacker.')
                elif cyber_agent_type == 'passive':
                    pass
                    logger.info(f'Simulation {sim_id}. Deploying passive attacker.')
                else:
                    raise ValueError(f'Simulation {sim_id}. Unknown attacker agent type: {cyber_agent_type}')
                logger.debug(f'Simulation {sim_id}. Step {step}: Now initiating attack.')

            # action = agent.sample_action(state=state)
            action = agent.compute_action_from_dict(state, infos["attacker"]["action_mask"])
            # action is tuple! (0/1, ACTION)

            # obs, rewards, terminations, truncations, infos = myEnv.step(action)
            # action into dict! {"attacker": action}
            # terminatoins, truncations are boolean dicts
            # obs is a dict degined in observation_space()
            # reward is a similar dict
            actions = {'attacker': action, 'defender': (0, None)}
            state, rewards, terminations, truncations, infos = myEnv.step(actions)

            total_reward += rewards['attacker']

            new_logs = torch.from_numpy(state['defender']['observed_state'])

            logger.debug(f"new_state: {new_logs}")
            logger.debug(f"old logs: {log_line}")


            if log_line is None:
                log_line = new_logs
            else:
                log_line = ((log_line == 0) & (new_logs == 1)).to(torch.long)

            logger.debug(f"new logs: {log_line}")

            flips = torch.bernoulli(torch.full_like(log_line.to(torch.float), 0.1)) * (log_line == 0)

            log_line = (log_line + flips).to(torch.long)

            logger.debug(f"fin logs: {log_line}")

            logger.debug(f"=============================")

            log_feature_vectors = torch.cat((log_feature_vectors[:, 1:], log_line.unsqueeze(1)), dim=1)

            if terminations['attacker']:
                logger.info(f"All steps compromised ({step} - {start_step} = {step-start_step})")
                if log_steps_after_total_compromise == 0:
                    logger.debug(f'Simulation {sim_id}. Step {step}: All attack steps were compromised after {step_start_step} steps. The graph contains {n_nodes} attack steps. Continuing to log for {max_log_steps_after_total_compromise} steps.')
                log_steps_after_total_compromise += 1
                if log_steps_after_total_compromise > max_log_steps_after_total_compromise:
                    logger.debug(f'Simulation {sim_id} terminated due to complete compromise.')
                    break

            nodes_as_features = torch.tensor(list(state['defender']['step_name'])).unsqueeze(1)

            combined_features = torch.cat((nodes_as_features, log_feature_vectors), dim=1).to(torch.long)
            edge_index = torch.from_numpy(state['defender']['edges'].T).to(torch.long)
            edge_type = torch.zeros(edge_index.shape[1]).to(torch.long)
            labels = torch.from_numpy(state['attacker']['observed_state']).to(torch.long)
            labels = torch.where(labels==-1, torch.tensor(0), labels)

            snapshot = Data(x=combined_features, edge_index=edge_index, edge_type=edge_type, y=labels)

            # Only add snapshots after the log window has been filled with unmalicious log lines
            if step >= log_window:
                snapshot_sequence.append(snapshot)

            if truncations['attacker']:
                logger.debug(f'Simulation {sim_id} terminated by PyRDDLGym.')
                break
        logger.debug(f'Simulation {sim_id} ended after {step} steps. Game time was set to {myEnv.horizon}.')
        myEnv.close()
        del myEnv
        end_time = time.time()
        # TODO see if i need to keep track of the attack graph
        # indexed_snapshot_sequence = {'snapshot_sequence': snapshot_sequence, 'graph_index': graph_indexe
        indexed_snapshot_sequence = {'snapshot_sequence': snapshot_sequence}
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        output_file = f"{storage_path}log_window_{log_window}/{n_nodes}_nodes/{len(snapshot_sequence)}_snapshots/{cyber_agent_type}/{date_time_str}.pkl"
        buffer = io.BytesIO()
        torch.save(indexed_snapshot_sequence, buffer)
        os.makedirs("data/" + os.path.dirname(output_file), exist_ok=True)
        with open(f"data/{output_file}", "wb") as f:
            f.write(buffer.getvalue())
        logger.info(f"Wrote training data to {output_file}")

        # buffer.seek(0)
        # blob = bucket.blob(output_file)
        # modified_retry = DEFAULT_RETRY.with_deadline(60)
        # modified_retry = modified_retry.with_delay(initial=0.5, multiplier=1.2, maximum=10.0)
        #
        # blob.upload_from_file(buffer, retry=modified_retry)
        # buffer.close()
        logger.info(f'Simulation {sim_id} completed. Time: {end_time - start_time:.2f}s. Written to {output_file}.')
        return output_file

    def produce_training_data_parallel(
        self,
        bucket_name,
        domain_rddl_path,
        instance_rddl_filepaths,
        log_window=25,
        max_start_time_step=100,
        max_log_steps_after_total_compromise=50,
        rddl_path='rddl/',
        snapshot_sequence_path = 'snapshot_sequences/',
        agent_type='random',
        novelty_priority=2,
        random_agent_seed=None):

        start_time = time.time()

        # local_domain_filepath = f'/tmp/domain.rddl'
        # storage_client = storage.Client()
        # bucket = storage_client.get_bucket(bucket_name)
        # domain_blob = bucket.blob(domain_rddl_path)
        # domain_blob.download_to_filename(local_domain_filepath)
        local_domain_filepath = domain_rddl_path



        n_simulations = len(instance_rddl_filepaths)
        n_processes = multiprocessing.cpu_count()
        result_filenames = []
        logger.info(f'Starting simulation of {n_simulations} instance models and a log window of {log_window}.')
        pool = multiprocessing.Pool(processes=n_processes)

        simulation_args = [(i, bucket_name, log_window, max_start_time_step,
                            max_log_steps_after_total_compromise,
                            local_domain_filepath, instance_rddl_filepaths[i],
                            snapshot_sequence_path, agent_type,
                            novelty_priority, random_agent_seed) for i in
            range(n_simulations)]

        result_filenames = pool.starmap(self.simulation_worker, simulation_args)
        pool.close()
        pool.join()

        logger.info(f'Sapshot sequence data generation completed. Time: {time.time() - start_time:.2f}s.')
