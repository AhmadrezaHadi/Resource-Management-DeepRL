from enum import Enum
from turtle import back
from typing import Counter
from xxlimited import new
from colorama import Back
import numpy as np
import matplotlib.pyplot as plt
from job import Job
from parameters import env_params
from agent import Agent
from backlog import Backlog
from data_generator import DataGenerator
# from logger import Logger, LogLevel



class TerminationType(Enum):
    NoNewJob = 1
    AllJobsDone = 2

class Environment():
    """
    Environment of the agent
    """
    def __init__(self, parameters, to_render=False, termination_type=TerminationType.NoNewJob) -> None:
        self.seq_nummber = 0
        self.seq_idx = 0                    # index in the current example sequence
        self.current_time = 0               # Current time of the machine
        self.current_queue_size = 0     

        self.to_render = False
        self.termination_type = termination_type

        self.backlog_size = parameters['backlog_size']                  # Size of the backlog
        self.hold_penalty = parameters.hold_penalty                     # hold penalty, used if we want to penalize the jobs in the queue
        self.delay_penalty = parameters.delay_penalty                   # delay penalty, used to penalize the jobs in the machine
        self.work_queue_size = parameters.work_queue_size               # size of the work queue
        self.job_queue = np.full(self.work_queue_size, None)            # job queue generated by work_queue_size
        self.simulation_length = parameters.simulation_length           # number of sequences
        self.episode_max_length = parameters.episode_max_length         # length of the episode
        self.job_sequence_length = parameters.job_sequence_length       # sequence of the job length
        

        self.data_generator = DataGenerator(parameters=parameters)      # data generator for new sequences
        self.agent = Agent(resources_count=parameters['number_resources'],      # our agent
            slots_count=parameters['max_resource_slots'], 
            time_horizon=parameters['time_horizon'])
        self.backlog = Backlog(backlog_size=parameters['backlog_size'])         # backlog of the environment

        self.work_sequences = self.generate_work_sequences()                    # work sequences
        self.actions = range(self.work_queue_size + 1)                          # actions which can be taken, 0-n-1 schedule job at that indes +1 for the empty action
    
    def step(self, action):
        """
        Performs a step in the environment. If the allocation was successful and the backlog is not empty
        moves the first backlog job into the working queue. If the allocation was not successful the next
        job from the sequence will either be added to the working queue if there is available slot
        or to the backlog otherwise.
        Returns state, reward, done triplet.
        """
        reward = 0
        done = False
        allocation = False

        if action < len(self.job_queue) and self.job_queue[action] is not None:
            allocation = self.agent.schedule_job(job=self.job_queue[action], current_time=self.current_time)
        
        if allocation:
            # remove job from queue
            self.job_queue[action] = None
            self.current_queue_size -= 1
            # fill job queue with backlog
            if not self.backlog.empty():
                self.job_queue[action] = self.backlog.dequeue()
                self.current_queue_size += 1 
        else:
            self.current_time += 1
            self.agent.proceed_time(current_time=self.current_time)

            # check whether to proceed
            done = self.done()
            reward = self.reward()

            if not done:
                self.fill_queue_and_backlog()
        
        # state = self.return state
        return state, reward, done, allocation 
    
    def reward(self):
        """
        Calculates the reward of the environment. The reward is the sum of -1/T_j where
        T_j is the length of each job in the system either currently scheduled, 
        in the job_queue waiting to be scheduled or in the backlog.
        """
        reward = self.agent.calc_delay_panalty(self.delay_penalty)
        
        for job in self.job_queue:
            if job is not None:
                reward += self.hold_penalty / float(job.length)
        reward += self.backlog.calc_panalty(self.dismiss_penalty) # uncomment to add the reward from the backlog
        return reward
    
    def done(self):
        """
        Checks if the environment has anymore valid actions to be 
        performed based on the termination type. For training the 
        termination type should be AllJobsDone i.e. all jobs have 
        finished executing or the time_step exceeds the episode length.
        """
        if self.termination_type == TerminationType.NoNewJob:
            return self.seq_idx >= self.job_sequence_length # current sequence is completed
        elif self.termination_type == TerminationType.AllJobsDone:
            if self.seq_idx >= self.job_sequence_length and         \
                len(self.agent.running_jobs) == 0 and               \
                all(slot is None for slot in self.job_queue) and    \
                self.backlog.empty():

                return True
            elif self.current_time > self.episode_max_length:
                return True
        return False
    
    def render(self):
        pass

    def fill_queue_and_backlog(self):
        """
        Fill up the queue and backlog.
        """
        if self.done():
            return
        # fill queue from backlog
        while not self.backlog.empty() and self.current_queue_size < self.work_queue_size:
            job = self.backlog.dequeue()
            idx = np.where(self.job_queue == None)[0][0]
            self.job_queue[idx] = job
            self.current_queue_size += 1
        
        # fill queue from simulation (if backlog is empty or not enough)
        while self.current_queue_size < self.work_queue_size and self.seq_idx < self.job_sequence_length:
            new_job = self.work_sequences[self.seq_nummber, self.seq_idx]
            if new_job is not None:
                new_job.set_enter_time(self.current_time)
                idx = np.where(self.job_queue == None)[0][0]
                self.job_queue[idx] = new_job
                self.current_queue_size += 1
            self.seq_idx += 1

        # fill backlog from simulation
        while self.seq_idx < self.job_sequence_length:
            new_job = self.work_sequences[self.seq_nummber, self.seq_idx]
            if new_job is not None:
                if self.backlog.enqueue(new_job) == True:
                    new_job.set_enter_time(self.current_time)
                else: 
                    break
            self.seq_idx += 1
            
    
    def generate_work_sequences(self):
        """
        Generates work sequences
        Output: @simulation_length x @job_sequence_length array of type Job
        """
        counter = 1
        work_sequences = np.full((self.simulation_length, self.job_sequence_length), None, dtype=object)
        for i in range(self.simulation_length):
            job_lengths, job_resource_vectors = self.data_generator.generate_sequence()
            for j in range(self.job_sequence_length):
                if job_lengths[j] > 0:
                    work_sequences[i, j] = Job(job_resource_vectors[j], job_lengths[j], id=counter)
                    counter += 1
        return work_sequences

    def print_work_sequence(self):
        """
        Print the current work sequence.
        """
        for x in self.work_sequences:
            for job in x:
                if job is not None:
                    print(str(job))
                else:
                    print("Job is None")

    def retrieve_state(self):
        pass
        
    def reward_completion(self):
        """
        Reward to optimize for completion time. -|J|, where J is the 
        jobs currently in the system.
        """
        reward = len(self.agent.running_jobs)
        for job in self.job_queue:
            if job is not None:
                reward += 1
        reward += self.backlog.num_jobs # uncomment to add the reward from the backlog

        return -reward


    def reset(self):
        """
        Resets the environment into it's initial state.
        """
        self.current_queue_size = 0
        self.seq_idx = 0
        self.current_time = 0
        self.agent.reset()
        self.job_queue = np.full(self.work_queue_size, None)
        self.backlog = Backlog(self.backlog_size)

        # The queue and backlog should be filled up beforehand
        self.fill_queue_and_backlog()