import numpy as np


class Job:
    '''
    Class for a single job in the system
    '''
    def __init__(self, resource_vector, length, id) -> None:
        self.resource_vector = resource_vector
        self.length = length
        self.id = id
        self.enter_time = -1
        self.start_time = -1
        self.finish_time = -1

    def set_start_time(self, start_time):
        self.start_time = start_time 
        self.finish_time = self.start_time + self.length
    
    def set_enter_time(self, time):
        self.enter_time = time
    
    def __eq__(self, __o: object) -> bool:
        return self.length == __o.length and \
            (self.resource_vector == __o.resource_vector).all() \
                and self.enter_time == __o.enter_time and \
                self.start_time == __o.start_time and \
                self.finish_time == __o.finish_time
    
