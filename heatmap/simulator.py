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
from malsim.agents.searchers import BreadthFirstAttacker, PassiveAttacker, DepthFirstAttacker
from maltoolbox.language import specification, LanguageGraph, LanguageClassesFactory
from maltoolbox.attackgraph import AttackGraph
from maltoolbox.model import Model
from malsim.sims.mal_simulator import MalSimulator
from malsim.agents.searchers import BreadthFirstAttacker

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
                          random_cyber_agent_seed=None,
                          sync_online=False,
                          fp_rate=0.1):
        import sys
        heatmap_logging.debug = '--debug' in sys.argv
        # logger = heatmap_logging.setup_logging()

        logger.info(f'Simulation {sim_id} started.')
#
        storage_client = storage.Client()
        # bucket = storage_client.get_bucket(bucket_name)
        # instance_blob = bucket.blob(instance_rddl_filepath)
        # local_instance_filepath = f'/tmp/instance_{sim_id}.rddl'
        # instance_blob.download_to_filename(local_instance_filepath)
        # instance_blob.delete()
        # gi_blob.delete()

        domain_rddl_path = "langs/heatmap.mal"
        try:
            lang_graph = LanguageGraph.from_mar_archive(domain_rddl_path)
        except:
            lang_graph = LanguageGraph.from_mal_spec(domain_rddl_path)
        lang_classes_factory = LanguageClassesFactory(lang_graph)
        # lang_classes_factory.create_classes()

        model = Model.load_from_file("langs/instance-model.yml", lang_classes_factory)
        model.save_to_file("tmp/model.yml")

        attack_graph = AttackGraph(lang_graph, model)
        attack_graph.attach_attackers()
        attack_graph.save_to_file("tmp/attack_graph.json")

        # max_iter is the episode horizon
        horizon = 250
        myEnv = MalSimulator(lang_graph, model, attack_graph, max_iter=horizon)

        start_time = time.time()
        n_nodes = len(attack_graph.nodes)
        # start_step = random.randint(log_window, max_start_time_step)
        start_step = random.randint(log_window, 100)

        logger.info(f"start step: {start_step}")

        agent = PassiveAttacker()
        snapshot_sequence = []
        log_feature_vectors = torch.zeros((n_nodes, log_window))
        myEnv.horizon = horizon
        log_steps_after_total_compromise = log_window

        myEnv.register_attacker("attacker", 0)
        myEnv.register_defender("defender")

        state, infos = myEnv.reset()
        labels = None
        log_line = None
        old_state = None
        nodes_as_features = None


        for step in range(myEnv.horizon):
            if step % 30 == 0:
                logger.info(f'Simulation {sim_id}. Step {step}/{myEnv.horizon}. Time: {time.time() - start_time:.2f}s.')
            if step == start_step:
                if cyber_agent_type == 'random':
                    # agent = BreadthFirstAttacker({"randomize": True})
                    agent = DepthFirstAttacker({"randomize": True})
                    logger.info(f'Simulation {sim_id}. Deploying random attacker at step {step}.')
                elif cyber_agent_type == 'passive':
                    pass
                    logger.info(f'Simulation {sim_id}. Deploying passive attacker at step {step}.')
                else:
                    raise ValueError(f'Simulation {sim_id}. Unknown attacker agent type: {cyber_agent_type}')
                logger.debug(f'Simulation {sim_id}. Step {step}: Now initiating attack.')

            try:
                action = agent.compute_action_from_dict(state, infos["attacker"]["action_mask"])
            except KeyError:
                pass

            actions = {'attacker': action, 'defender': (0, None)}
            state, rewards, terminations, truncations, infos = myEnv.step(actions)

            if not 'attacker' in state:
                state, rewards, terminations, truncations, infos = _state, _rewards, _terminations, _truncations, _infos

            try:
                new_state = torch.from_numpy(state['defender']['observed_state'])
            except TypeError:
                new_state = torch.Tensor(state['defender']['observed_state'])

            if old_state is None:
                new_state = torch.full_like(new_state.to(torch.long), 0).to(torch.long)
                old_state = new_state

            logger.debug(f"new_state: {new_state}")
            logger.debug(f"old logs: {log_line}")

            if log_line is None:
                log_line = new_state
            else:
                log_line = ((log_line == 0) & (new_state == 1) & (old_state == 1)).to(torch.long)
                log_line = ((log_line == 0) & (torch.logical_xor(new_state, old_state) == 1)).to(torch.long)

            logger.debug(f"new logs: {log_line}")

            flips = torch.bernoulli(torch.full_like(log_line.to(torch.float), fp_rate)) * (log_line == 0)

            log_line = (log_line + flips).to(torch.long)

            logger.debug(f"fin logs: {log_line}")

            old_state = new_state

            logger.debug(f"=============================")

            log_feature_vectors = torch.cat((log_feature_vectors[:, 1:], log_line.unsqueeze(1)), dim=1)


            if terminations['attacker']:
                if log_steps_after_total_compromise == log_window:
                    logger.info(f"All steps compromised ({step} - {start_step} = {step-start_step})")
                if log_steps_after_total_compromise == 0:
                    logger.debug(f'Simulation {sim_id}. Step {step}: All attack steps were compromised after {step_start_step} steps. The graph contains {n_nodes} attack steps. Continuing to log for {max_log_steps_after_total_compromise} steps.')
                log_steps_after_total_compromise -= 1
                if not log_steps_after_total_compromise:
                    logger.debug(f'Simulation {sim_id} terminated due to complete compromise.')
                    logger.info(f'END: {step}')
                    break

            if nodes_as_features is None:
                nodes_as_features = torch.tensor(list(state['defender']['step_name'])).unsqueeze(1)
                edge_index = torch.from_numpy(state['defender']['edges'].T).to(torch.long)
                edge_type = torch.zeros(edge_index.shape[1]).to(torch.long)

            combined_features = torch.cat((nodes_as_features, log_feature_vectors), dim=1).to(torch.long)
            if step < start_step-1:
                labels = torch.zeros(state['attacker']['observed_state'].shape).to(torch.long)
            elif cyber_agent_type != 'passive':
                labels = torch.from_numpy(state['attacker']['observed_state']).to(torch.long)
                labels = torch.where(labels==-1, torch.tensor(0), labels)
                labels =labels.to(torch.long)

            attacker_state = state['attacker']['observed_state']
            snapshot = Data(x=combined_features, edge_index=edge_index,
                            edge_type=edge_type, y=labels)
            # Only add snapshots after the log window has been filled with unmalicious log lines
            if step >= log_window:
                snapshot_sequence.append(snapshot)

            if truncations['attacker']:
                logger.debug(f'Simulation {sim_id} terminated by PyRDDLGym.')
                break

            _state, _rewards, _terminations, _truncations, _infos = state, rewards, terminations, truncations, infos
        logger.debug(f'Simulation {sim_id} ended after {step} steps. Game time was set to {myEnv.horizon}.')
        myEnv.close()
        end_time = time.time()
        indexed_snapshot_sequence = {'snapshot_sequence': snapshot_sequence}
        date_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
        output_file = f"{storage_path}log_window_{log_window}/{n_nodes}_nodes/fpr_{fp_rate}/{len(snapshot_sequence)}_snapshots/{cyber_agent_type}/{date_time_str}.pkl"
        buffer = io.BytesIO()
        torch.save(indexed_snapshot_sequence, buffer)
        os.makedirs("data/" + os.path.dirname(output_file), exist_ok=True)
        with open(f"data/{output_file}", "wb") as f:
            f.write(buffer.getvalue())
        logger.info(f"Wrote training data to {output_file}")

        if sync_online:
            buffer.seek(0)
            blob = bucket.blob(output_file)
            modified_retry = DEFAULT_RETRY.with_deadline(60)
            modified_retry = modified_retry.with_delay(initial=0.5, multiplier=1.2, maximum=10.0)
            blob.upload_from_file(buffer, retry=modified_retry)

        del myEnv
        buffer.close()

        logger.info(f'Simulation {sim_id} completed. Time: {end_time - start_time:.2f}s. Written to {output_file}.')
        return output_file

    def produce_training_data_parallel(
        self,
        bucket_name,
        domain_rddl_path,
        instance_rddl_filepaths,
        n_simulations=1,
        log_window=25,
        max_start_time_step=100,
        max_log_steps_after_total_compromise=50,
        rddl_path='rddl/',
        snapshot_sequence_path = 'snapshot_sequences/',
        agent_type='random',
        novelty_priority=2,
        random_agent_seed=None,
        sync_online=False,
        fp_rate=0.1):

        start_time = time.time()

        # local_domain_filepath = f'/tmp/domain.rddl'
        # storage_client = storage.Client()
        # bucket = storage_client.get_bucket(bucket_name)
        # domain_blob = bucket.blob(domain_rddl_path)
        # domain_blob.download_to_filename(local_domain_filepath)
        local_domain_filepath = domain_rddl_path



        # n_simulations = len(instance_rddl_filepaths)
        n_processes = multiprocessing.cpu_count()
        result_filenames = []
        logger.info(f'Starting simulation of {n_simulations} instance models and a log window of {log_window}.')
        pool = multiprocessing.Pool(processes=n_processes)

        simulation_args = [(i, bucket_name, log_window, max_start_time_step,
                            max_log_steps_after_total_compromise,
                            local_domain_filepath, instance_rddl_filepaths[0],
                            snapshot_sequence_path, agent_type if i%3 != 2 else 'passive',
                            novelty_priority, random_agent_seed, sync_online, fp_rate) for i in
            range(n_simulations)]
        result_filenames = pool.starmap(self.simulation_worker,
        simulation_args)
        pool.close()
        pool.join()

        # result_filenames = [self.simulation_worker(0,  bucket_name, log_window,
                                                   # max_start_time_step,
                            # max_log_steps_after_total_compromise,
                            # local_domain_filepath, instance_rddl_filepaths[0],
                            # snapshot_sequence_path, agent_type,
                            # novelty_priority, random_agent_seed, sync_online, fp_rate)]
#
        logger.info(f'Sapshot sequence data generation completed. Time: {time.time() - start_time:.2f}s.')
