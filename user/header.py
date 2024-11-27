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



class ForwardEulerPEM(nn.Module):  # use steps or R2 as switch

    def __init__(self, model, factor, dt, N, update, threshold1=0, threshold2=0,
                 sensitivity=500,  train=2000):  # sensitivity=100

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






