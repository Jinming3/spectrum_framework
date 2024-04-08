"""
contains functions for NN
"""
import copy
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import torch.nn as nn
from torch.jit import Final
from typing import List, Tuple, Any
from pem import PEM

def R2(Y_sys, Yhat):
    """
    accuracy of a model
    Y_sys: real measurement in an array
    Yhat: model prediction
    """
    s1 = np.sum((Y_sys - Yhat) ** 2)
    mean = np.mean(Y_sys)
    s2 = np.sum((Y_sys - mean) ** 2)
    return 1.0 - s1 / s2

def mse(Y_sys, Yhat):  # used in class ForwardEulerPEM_ahead
    s = np.sum((Y_sys - Yhat) ** 2)
    m = s / len(Y_sys)
    return m

#  #------------- torch original >>>>>> -----------
class MechanicalSystem(nn.Module):  # original

    def __init__(self, dt, n_x=2, init_small=True):
        super(MechanicalSystem, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = 64
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1))

        if init_small:
            for i in self.net.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)

    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        in_xu = torch.cat((x1, u1), -1)
        dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class ForwardEuler(nn.Module):  # original

    def __init__(self, model, dt):
        super(ForwardEuler, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        xhat_list = list()
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx = self.model(x_step, u_step)
            x_step = x_step + dx * self.dt
            xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)
        return xhat


class CascadedTanksOverflowNeuralStateSpaceModel(nn.Module):

    def __init__(self, n_feat=100, scale_dx=1.0, init_small=True):
        super(CascadedTanksOverflowNeuralStateSpaceModel, self).__init__()
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        # Neural network for the first state equation = NN(x_1, u)
        self.net_dx1 = nn.Sequential(
            nn.Linear(2, n_feat),
            nn.ReLU(),
            # nn.Linear(n_feat, n_feat),
            # nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Neural network for the first state equation = NN(x_1, x2, u) # we assume that with overflow the input may influence the 2nd tank instantaneously
        self.net_dx2 = nn.Sequential(
            nn.Linear(3, n_feat),
            nn.ReLU(),
            # nn.Linear(n_feat, n_feat),
            # nn.ReLU(),
            nn.Linear(n_feat, 1),
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx1.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net_dx2.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):

        # the first state derivative is NN_1(x1, u)
        in_1 = torch.cat((in_x[..., [0]], in_u), -1)  # concatenate 1st state component with input
        dx_1 = self.net_dx1(in_1)

        # the second state derivative is NN_2(x1, x2, u)
        in_2 = torch.cat((in_x, in_u), -1)  # concatenate states with input to define the
        dx_2 = self.net_dx2(in_2)

        # the state derivative is built by concatenation of dx_1 and dx_2, possibly scaled for numerical convenience
        dx = torch.cat((dx_1, dx_2), -1)
        dx = dx * self.scale_dx
        return dx





class ForwardEulerPEM(nn.Module):  # use steps or R2 as switch

    def __init__(self, model, factor, dt, N, update, threshold1=0, threshold2=0,
                 sensitivity=600,  train=2000, param=np.array):  # sensitivity=100

        super(ForwardEulerPEM, self).__init__()
        self.factor = factor
        self.model = model
        self.dt = dt
        self.N = N

        self.update = update  # choose case

        self.train = train
        self.threshold1 = threshold1  # start update
        self.threshold2 = threshold2  # stop update
        self.sensitivity = sensitivity  # an sequence to monitor R2
        self.stop = []
        self.correction = []
        self.xhat_data = np.zeros((N, 2))
        self.param = [param]

    def forward(self, x0: torch.Tensor, u: torch.Tensor, y):
        x_step = x0
        self.Thehat = np.zeros((self.N, 6))

        self.y_pem = []
        self.y_pem0 = []
        self.r2 = np.zeros(self.N)
        self.err = np.zeros(self.N)  # |y-yhat|
        # ---------------
        q = 0
        while q < self.N:

            if self.update == 0: # not updating, no PEM, just basic inference in simple forward Euler
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()
                q = q + 1

            if self.update == 1:  # update non-stop:
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                self.xhat_data[q, :] = x_out
                x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                # match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                q = q + 1

            if self.update == 5:  # update with threshold,  adding resting PEM !! # use this!!
                u_step = u[q]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)# non-updating pem added
                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect
                self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                # # --------------------------------------------------------
                # if q < 1000:
                #     y_nn = x_step[:, 0].clone().detach().numpy()
                #     u_in = y_nn
                #     self.factor.pem_one(y[q] - y_nn, u_in, on=True)
                #     x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                #     self.xhat_data[q, :] = x_out  # collect
                #     x_step = torch.tensor(x_out, dtype=torch.float32)  # ! update input to NN !
                #
                # # --------------------------------------------------------
                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])


                self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                # match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0]) # check the dimension before use
                match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # if q > self.sensitivity:
                #     match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                # if q <= self.sensitivity:
                #     match = R2(y[0:q, 0, 0], self.xhat_data[0:q, 0])
                match = round(match, 3)
                self.r2[q] = match
                if match < self.threshold1:
                    # print(f'update at {q}, with R2= {match}')
                    self.correction.append([match, q])
                    while q < self.N:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()
                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        self.err[q] = y[q] - x_step[0, 0].clone().detach().numpy()
                        x_out = x_step.clone().detach().numpy() + self.factor.Xhat[:, 0]
                        self.xhat_data[q, :] = x_out
                        x_step = torch.tensor(x_out, dtype=torch.float32)  # don't delete this ! update input to NN !
                        # match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])

                        # if q > self.sensitivity:
                        #     match = R2(y[q - self.sensitivity:q, 0, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        # if q <= self.sensitivity:
                        #     match = R2(y[0:q, 0, 0], self.xhat_data[0:q, 0])
                        match = round(match, 3)
                        self.r2[q] = match
                        if match > self.threshold2:
                            self.stop.append([match, q])
                            # print(f'finish at  {q}, with R2= {match}')
                            break
                        q = q + 1

                y_nn = x_step[:, 0].clone().detach().numpy()

                self.factor.pem_one(y[q-1] - y_nn, y_nn, on=False)  # for pem n-step ahead, y[q]

                q = q + 1

            if self.update == 8: # xtep = x_step+pem, copied from 5, use self.train to stop PEM, not R2,  added when stop
                u_step = u[q]
                dx = self.model(x_step, u_step)

                x_step = x_step + dx * self.dt + torch.tensor(self.factor.Xhat[:, 0], dtype=torch.float32)# non-updating pem added

                self.xhat_data[q, :] = x_step[0, :].clone().detach().numpy()  # collect

                self.y_pem0.append([self.factor.Xhat[0, 0], q])
                self.y_pem.append([None, q])

                if q > self.sensitivity:
                    match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                if q <= self.sensitivity:
                    match = R2(y[0:q, 0], self.xhat_data[0:q, 0])
                self.r2[q] = match
                while q < self.train:
                        u_step = u[q]
                        dx = self.model(x_step, u_step)
                        x_step = x_step + dx * self.dt
                        y_nn = x_step[:, 0].clone().detach().numpy()
                        self.factor.pem_one(y[q] - y_nn, y_nn, on=True)
                        self.y_pem.append([self.factor.Xhat[0, 0], q])
                        self.y_pem0.append([None, q])
                        self.Thehat[q, :] = np.copy(self.factor.Thehat[:, 0])
                        x_out = x_step.clone().detach().numpy()+ self.factor.Xhat[:, 0]
                        self.xhat_data[q, :] = x_out
                        x_step = torch.tensor(x_out, dtype=torch.float32)  # don't delete this ! update input to NN !
                        match = R2(y[q - self.sensitivity:q, 0], self.xhat_data[q - self.sensitivity:q, 0])
                        self.r2[q] = match
                        q = q + 1
                y_nn = x_step[:, 0].clone().detach().numpy()
                self.factor.pem_one(y[q] - y_nn, y_nn, on=False)  # for pem n-step ahead
                q = q + 1

        print('stop at', self.stop)
        print('update at', self.correction)
        return self.xhat_data






