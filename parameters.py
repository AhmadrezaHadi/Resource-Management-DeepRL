import math


# Parameters 
t = 3
r = 50
time_horizon = 20*t
job_seq_len = 100
sim_len = 5
backlog_size = 60



# Environment parameters 
env_params = {
    't' : t,
    'r' : r,
    'time_horizon' : time_horizon,
    'job_sequence_length' : job_seq_len,
    'simulation_length' : 5,
    'number_resources' : 2,
    'job_rate' : 0.70,
    'input_height' :  time_horizon,
    'backlog_size' : backlog_size,
    'max_resource_slots' : r,
    'work_queue_size' : 10 ,

    'backlog_width' : \
        int(math.ceil(self.backlog_size / float(self.time_horizon)))
    'input_width' : self.network_input_width : \
        int(1 + self.work_queue_size) * self.number_resources + self.backlog_width # the network input width
            

    'dismiss_penalty' : -1,
    'hold_penalty' : -1,
    'delay_penalty' : -1,
    'episode_max_length' : 150                                      
}