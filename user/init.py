import numpy as np
import torch


# -------- all trianing settings are here-------
class ModelTrain_setting:
    """
    the only place for users to change the training parameters, universally, unless redefined before calling a function
    """

    def __init__(self, sys_name):
        self.ahead_step_range=np.ones((1, 2))
        # self.ahead_step_range=np.ones((1, 2), dtype=int)
        # self.ahead_step_range = np.ones(2, dtype=int)
        self.sys_name = sys_name
        self.module = 'module_name'


        if sys_name == 'EMPS':
            self.ts = 0.005
            self.time_train = 10  # seconds, time for generating training data
            self.time_spectrum_train = 120
            self.change_u = np.array([20,40, 60,80, 100]) # change the spectrum in u, theta not change
            self.pause = 40
            self.time_test = 50  # seconds, time for generating testing data
            self.change = np.array([20, 50]) # aging

            self.dt = torch.tensor(self.ts, dtype=torch.float32)



        elif sys_name == 'tanks':
            self.ts = 4.0
            self.change = np.array([1000, 2000])
            self.time_train = 4096
            self.time_test = 4096
            # for modeling
            self.batch_num = 64
            self.batch_length = 32  #128
            self.lr = 0.0001
            self.n_x = 2
            self.hidden = 64
            self.dt = torch.tensor(self.ts, dtype=torch.float32)



        elif sys_name == 'springs':
            self.ts = 0.05
            self.time_train = 100
            self.time_spectrum_train = 100
            self.change_u = np.array([20, 40, 55, 70,80, 90]) # change the spectrum in u, theta not change
            self.time_test = 150
            self.change = np.array([20, 40, 55, 70, 100, 125]) #aging
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
