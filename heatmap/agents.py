import random
import logging
from collections import OrderedDict
from pyRDDLGym.core.policy import BaseAgent

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

class LessRandomCyberAgent(BaseAgent):
    def __init__(self, action_space, novelty_priority=2, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.weighted_actions = dict()
        self.novelty_priority = 1 # Newly discovered actions are five times as likely to be chosen

    def action_horizon(self, state):
        actionable_items = []
        for key, value in state.items():
            if key.startswith("available_for_") and value:
                action_key = key.replace("available_for_", "")
                actionable_items.append(action_key)
        return actionable_items

    def sample_action(self, state=None):
        possible_actions = self.action_horizon(state)
        # Remove any actions that are no longer possible
        self.remove_obsolete_actions(possible_actions)
        new_possible_actions = [action for action in possible_actions if action not in self.weighted_actions]
        if new_possible_actions:
            self.add_new_actions(new_possible_actions)
        action = {}
        if len(self.weighted_actions) > 0:
            action = self.choose_action(action)
        else:
            s = self.action_space.sample()
            action[list(s.keys())[0]] = 0
        return action

    def choose_action(self, action):
        keys = list(self.weighted_actions.keys())
        weights = list(self.weighted_actions.values())
        selected_action = random.choices(keys, weights, k=1)[0]
        action[selected_action] = 1
        return action

    def add_new_actions(self, new_possible_actions):
        if len(self.weighted_actions) == 0:
            new_probability_weight = 1
        else:
            new_probability_weight = self.novelty_priority * max(self.weighted_actions.values())
        for action in new_possible_actions:
            self.weighted_actions[action] = new_probability_weight
        total_weight = sum(self.weighted_actions.values())
        self.weighted_actions = {action: weight / total_weight for action, weight in self.weighted_actions.items()}

    def remove_obsolete_actions(self, possible_actions):
        self.weighted_actions = {action: weight for action, weight in self.weighted_actions.items() if action in possible_actions}
        total_weight = sum(self.weighted_actions.values())
        self.weighted_actions = {action: weight / total_weight for action, weight in self.weighted_actions.items()}

class NoveltyFocusedRandomCyberAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)
        self.previous_possible_actions = []
        self.previous_selected_action = None
        self.repeat_action_probability = 0.5

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
        new_actions = []
        if len(possible_actions) > len(self.previous_possible_actions):
            new_actions = [pa for pa in possible_actions if pa not in self.previous_possible_actions]
        action = {}
        if len(new_actions) > 0:
            selected_action = random.choice(new_actions)
            action[selected_action] = 1
        elif self.previous_selected_action is not None and self.rng.random() < self.repeat_action_probability:
            selected_action = self.previous_selected_action
            action[selected_action] = 1
        elif len(possible_actions) > 0:
            selected_action = random.choice(possible_actions)
            action[selected_action] = 1
        else:
            action[list(s.keys())[0]] = 0
        self.previous_possible_actions = possible_actions
        self.previous_selected_action = selected_action
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
