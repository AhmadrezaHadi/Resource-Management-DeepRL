from job import Job
import numpy as np


class Agent:
    def __init__(self, resources_count, slots_count, time_horizon) -> None:
        self.resources_count = resources_count
        self.slots_count = slots_count
        self.time_horizon = time_horizon

        self.job_list = []
