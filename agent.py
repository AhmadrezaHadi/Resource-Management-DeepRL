from matplotlib.style import available
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
        """
        Resets machine to initial state.
        """
        self.running_jobs = []
        self.available_slots = np.full((self.time_horizon, self.resources_count), 
                                        self.slots_count)


    def schedule_job(self, job : Job, current_time, debug=False):
        """
        Tries to allocate a job if in the machine. The job must be with length not greater than the 
        observable time steps in the machine. If there are not enough resources currently in the machine 
        i.e. the job was not successfully allocated False will be returned, otherwise True.
        Future improvement can be to add job fragmentation.
        """
        # schedule jobs not at the same time on both resources

        # if ((self.available_slots - job.resource_vector) > 0).any(axis=0).all():
        #     for index, col in enumerate(self.available_slots.T):
        #         idx = np.where(col >= job.resource_vector[index])[0][0]
        #         self.available_slots[idx, index] -= job.resource_vector[index]
        #         if debug:
        #             # TODO compelete debug
        #             pass
        #     job.set_start_time(current_time)

        if ((self.available_slots - job.resource_vector) > 0).any(axis=0).all():
            for t, row in enumerate(self.available_slots):
                if (row - job.resource_vector >= 0).all():
                    # Update slots
                    row -= job.resource_vector
                    # Set start time
                    job.set_start_time(current_time + t)
                    # add to running jobs
                    self.running_jobs.append(job)
                    
                    break
        return False
    
    def proceed_time(self, current_time):
        """
        Proceed time of the machine 1 step forward
        """
        self.available_slots[:-1, :] = self.available_slots[1:, :]
        self.available_slots[-1, :] = self.slots_count

        for job in self.running_jobs:
            if job.finish_time <= current_time:
                self.running_jobs.remove(job)
                # self.logger.debug(f"Job {str(job)} finished at {current_time}")
    
    
