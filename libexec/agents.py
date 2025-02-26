"""
Custom agents.
"""

import json
import logging
import pprint
import random

from malsim.mal_simulator import MalSimAgentStateView
from malsim.agents import (
    BreadthFirstAttacker as BuiltinBFS,
)

logger = logging.getLogger(__name__)


class BreadthFirstAttacker(BuiltinBFS):
    default_settings = BuiltinBFS.default_settings | {"wait_factor": 0}

    def __init__(self, agent_config):
        super().__init__(agent_config)
        self.logs: list[dict] = []
        # TODO read this from the state instead
        self.attack_graph = agent_config.pop("attack_graph")

    def get_next_action(self, agent_state: MalSimAgentStateView, **kwargs):
        self._update_targets(agent_state.action_surface)

        act = random.choices(
            [True, False],
            weights=[1 - self.settings["wait_factor"], self.settings["wait_factor"]],
        )

        if act:
            self._select_next_target()
        else:
            self.current_target = None

        if self.current_target:
            self._collect_logs(agent_state, timestamp=kwargs["timestamp"])

        return self.current_target

    def _collect_logs(self, state, timestamp):
        attack_step = self.attack_graph.nodes[self.current_target.id]
        for _, detector in attack_step.detectors.items():
            log = {
                "timestamp": state.timestamp,
                "_detector": detector.name,
                "asset": str(attack_step.model_asset.name),
                "attack_step": attack_step.name,
                "agent": self.__class__.__name__,
                #'context': {},
            }

            for label, lgasset in detector.context.items():
                try:
                    *_, asset = (
                        step.model_asset
                        for step in self.attack_graph.attackers[0].reached_attack_steps
                        if step.model_asset.type
                        in [subasset.name for subasset in lgasset.sub_assets]
                    )
                except ValueError:
                    msg = (
                        f"Context {detector.context} cannot be satisfied "
                        f"for step {attack_step.full_name}. No {lgasset.name} "
                        "was compromised already."
                    )
                    raise ValueError(msg)

                log[label] = str(asset.name)

            self.logs.append(log)

            logger.info("Detector triggered on %s", attack_step.full_name)
            logger.info(pprint.pformat(log))

    def terminate(self):
        self._write_logs()

    def _write_logs(self):
        with open("logs.json", "w") as f:
            json.dump(self.logs, f, indent=2)
            self.logs = []


class DepthFirstAttacker(BreadthFirstAttacker):
    _extend_method = "extend"
