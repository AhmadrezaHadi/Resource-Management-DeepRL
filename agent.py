from job import Job
import numpy as np


class Agent:
    def __init__(self, resources_count, slots_count, time_horizon) -> None:
        self.resources_count = resources_count
        self.slots_count = slots_count
        self.time_horizon = time_horizon

        self.running_jobs = []
        self.available_slots = np.full((time_horizon, resources_count), slots_count)
    def reset(self):
        self.running_jobs = []
        self.available_slots = np.full((self.time_horizon, self.resources_count), 
                                        self.slots_count)


    def schedule_job(self, job : Job, current_time, debug=False):
        if ((self.available_slots - job.resource_vector) > 0).any(axis=0).all():
            for index, col in enumerate(self.available_slots.T):
                idx = np.where(col >= job.resource_vector[index])[0][0]
                self.available_slots[idx, index] -= job.resource_vector[index]
                if debug:
                    # TODO compelete debug
                    pass
            job.set_start_time(current_time)
        return False
