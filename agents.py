import random
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
        print(f'action = {action}')
        return action

class RandomCyberAgent(BaseAgent):
    def __init__(self, action_space, seed=None):
        self.action_space = action_space
        self.rng = random.Random(seed)
        if seed is not None:
            self.action_space.seed(seed)

    def sample_action(self, state=None):
        s = self.action_space.sample()
        action = {}
        selected_action = self.rng.sample(list(s), 1)[0]
        action[selected_action] = s[selected_action]
        print(f'RandomCyberAgent action = {action}')
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
