import numpy as np
import torch


# -------- all trianing settings are here-------
class ModelTrain_setting:
    """
    the only place for users to change the training parameters, universally, unless redefined before calling a function
    """

    def __init__(self, sys_name):

        if sys_name == 'tanks':
            self.ts = 4.0
            self.change = np.array([1000, 2000])
            self.time_train = 4096
            self.time_test = 4096
            # for model
            self.batch_num = 64
            self.batch_length = 32  #128
            self.lr = 0.0001
            self.n_x = 2
            self.hidden = 64
            self.dt = torch.tensor(self.ts, dtype=torch.float32)

        elif sys_name == 'EMPS':
            self.ts = 0.005
            self.change = np.array([20, 50])
            self.time_train = 20  # seconds, time for generating training data
            self.time_test = 70  # seconds, time for generating testing data
            # for model
            self.batch_num = 64
            self.batch_length = 32
            self.lr = 0.0001
            self.n_x = 2
            self.hidden = 64
            self.dt = torch.tensor(self.ts, dtype=torch.float32)

        elif sys_name == 'springs':
            self.ts = 0.05
            self.change = np.array([20, 40, 55, 70, 100, 125])
            self.time_train = 100
            self.time_test = 150
            # for model
            self.num_epoch = 10000
            self.batch_num = 64
            self.batch_length = 32
            self.lr = 0.0001
            self.hidden = 64
            self.dt = torch.tensor(self.ts, dtype=torch.float32)
            self.n_x = 2

        elif sys_name == 'rlc':
            self.ts = 0.5 * 10 ** (-6)  # for generate system
            self.change = np.array([0.5, 1, 2.5]) * 10 ** (-3)
            self.time_train = 2 * 10 ** (-3)
            self.time_test = 3 * 10 ** (-3)
            # for model
            self.num_epoch = 10000
            self.batch_num = 64
            self.batch_length = 64
            self.lr = 0.001
            self.hidden = 64
            self.n_x = 2
            self.n_u = 1
            self.dt = torch.tensor(1.0, dtype=torch.float32)  # for neural network state space

        else:
            raise Error("System not defined")
