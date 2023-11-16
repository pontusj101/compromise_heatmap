import time
import numpy
import random
from pyRDDLGym import RDDLEnv
from pyRDDLGym import ExampleManager
from pyRDDLGym.Visualizer.MovieGenerator import MovieGenerator
from pyRDDLGym.Core.Policies.Agents import RandomAgent
from agents import PassiveCyberAgent
from agents import RandomCyberAgent

base_path = 'content/'
myEnv = RDDLEnv.RDDLEnv(domain=base_path+'domain.rddl', instance=base_path+'instance.rddl')

start_step =  random.randint(10, 100)
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
    attacksteps = [key for key, value in next_state.items() if type(value) is numpy.bool_ and value == True and "observed" not in key]
    total_reward += reward
    print()
    print(f'step              = {step}')
    print(f'action            = {action}')
    print(f'observations      = {observations}')
    print(f'log trace (25)    = {log_trace[-25:]}')
    print(f'attack steps      = {attacksteps}')
    print(f'TTCs              = {[(attackstep, value) for attackstep, value in next_state.items() if type(value) is numpy.int64]}')
    print(f'reward            = {reward}')
    state = next_state
    if done:
        break
end_time = time.time()
print()
print(f'episode ended with reward {total_reward}. Execution time was {end_time-start_time} s.')

myEnv.close()