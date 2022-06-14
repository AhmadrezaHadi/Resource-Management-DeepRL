import numpy as np
from job import Job

class Backlog():
    """
    Class representing backlog queue
    """

    def __init__(self, backlog_size) -> None:
        self.size = backlog_size
        self.backlog = np.full(self.size, None)
        self.num_jobs = 0

    def empty(self):
        """
        returns true if the backlog is empty 
        """
        return self.num_jobs == 0
    
    def full(self):
        """
        returns true if the backlog is full
        """
        return self.num_jobs == self.size
    
    def enqueue(self, job:Job):
        """
        Tries to add job to the backlog if the backlog is not full
        return true if succesfull, false if not.
        """
        if self.num_jobs < self.size:
            self.backlog[self.num_jobs] = job
            self.num_jobs += 1
            return True
        return False

    def dequeue(self):
        """
        Removes job from backlog and returns it. If the backlog is empty None is returned.
        """
        if self.empty():
            return None

        if self.num_jobs > 0:
            job = self.backlog[0]
            self.backlog[:-1] = self.backlog[1:]
            self.backlog[-1] = None
            self.num_jobs -= 1
            return job
    
    