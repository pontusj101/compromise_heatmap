import random
import logging
from collections import OrderedDict
from pyRDDLGym.Core.Policies.Agents import BaseAgent

# Attackers and defenders look the same
class PassiveCyberAgent(BaseAgent):
    def __init__(self, action_space):
        self.action_space = action_space

    def sample_action(self, state=None):
        selected_action = next(iter(self.action_space.spaces))
        action = {selected_action: self.action_space[selected_action]}
        action[selected_action] = 0
        return action

class RandomCyberAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def action_horizon(self, state):
        actionable_items = []
        for key, value in state.items():
            if key.startswith("available_for_") and value:
                action_key = key.replace("available_for_", "")
                actionable_items.append(action_key)
        return actionable_items

    def sample_action(self, state=None):
        possible_actions = self.action_horizon(state)
        s = self.action_space.sample()
        action = {}
        if len(possible_actions) > 0:
            selected_action = random.choice(possible_actions)
            action[selected_action] = 1
        else:
            action[list(s.keys())[0]] = 0
        return action

class HostTargetedCyberAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def action_horizon(self, state):
        actionable_items = []
        for key, value in state.items():
            if key.startswith("available_for_") and value:
                action_key = key.replace("available_for_", "")
                actionable_items.append(action_key)
        return actionable_items

    def sample_action(self, state=None):
        s = self.action_space.sample()
        action = {}
        possible_actions = self.action_horizon(state)
        neighbor_hosts = [pa.split('___h')[1] for pa in possible_actions if 'compromise_attempt' in pa]
        for nh in neighbor_hosts:
            negibor_creds = [pa for pa in possible_actions if f'___c{nh}' in pa]
        if len(negibor_creds) > 0: 
            possible_actions = negibor_creds # if there are creds to neighbor hosts, try to crack them
        else:
            possible_actions = [pa for pa in possible_actions if 'compromise_attempt' in pa]  # otherwise, try to compromise a neighbor host
        if len(possible_actions) > 0:
            selected_action = random.choice(possible_actions)
            action[selected_action] = 1
        else:
            action[list(s.keys())[0]] = 0
        return action

class KeyboardCyberAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space

    def sample_action(self, state=None):
        available_actions = list(self.action_space.spaces.keys())

        print("Available actions:")
        for i, action in enumerate(available_actions):
            print(f"{i}. {action}")

        selected_index = int(input("Enter the index of the action you want to take: "))

        if selected_index < 0 or selected_index >= len(available_actions):
            print("Invalid index. Using a default action.")
            selected_index = 0

        selected_action = available_actions[selected_index]

        return {selected_action: 1}
