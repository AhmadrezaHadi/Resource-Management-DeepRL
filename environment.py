from enum import Enum
import numpy as np
import matplotlib.pyplot as plt
from parameters import env_params
# from logger import Logger, LogLevel


class TerminationType(Enum):
    NoNewJob = 1
    AllJobsDone = 2

class Environment():
    """
    Environment of the agent
    """
    def __init__(self, parameters,to_render=False, termination_type=TerminationType.NoNewJob) -> None:
        self.seq_nummber = 0
        self.seq_idx = 0
        self.current_time = 0
        self.current_queue_size = 0

        self.to_render = False
        self.termination_type = termination_type

        self.job_sequence_length = parameters.job_sequence_length
    
    
    def step(self, action):
        reward = 0
        done = False
        allocation = False

        if action < len(self.job_queue) and self.job_queue[action] is not None:
            allocation = 1
        
        if allocation:
            pass
        else:
            self.current_time += 1
            # self.machine.time_proceed(self.current_time)

            done = self.done()
            reward = self.reward()

            if not done:
                pass
        
        return state, reward, done, allocation 
    
    def reward(self):
        pass
    
    def done(self):
        pass
    
    def render(self):
        pass
