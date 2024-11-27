import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
import os
import sys
import math
from header import ForwardEulerPEM
import torch
import torch.nn as nn
from torch.jit import Final

np.random.seed(0)
torch.manual_seed(0)

class NeuralStateSpaceModel(nn.Module):  # when not pos and vel, no derivative relation, not used yet
    n_x: Final[int]
    n_u: Final[int]
    n_feat: Final[int]

    def __init__(self, n_x, n_u, n_feat, scale_dx, init_small=True, activation='relu'):
        super(NeuralStateSpaceModel, self).__init__()
        self.n_x = n_x
        self.n_u = n_u
        self.n_feat = n_feat
        self.scale_dx = scale_dx

        if activation == 'relu':
            activation = nn.ReLU()
        elif activation == 'softplus':
            activation = nn.Softplus()
        elif activation == 'tanh':
            activation = nn.Tanh()

        self.net = nn.Sequential(
            nn.Linear(n_x + n_u, n_feat),  # 2 states, 1 input
            activation,
            nn.Linear(n_feat, n_x)
        )

        # Small initialization is better for multi-step methods
        if init_small:
            for m in self.net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=1e-4)
                    nn.init.constant_(m.bias, val=0)

    def forward(self, in_x, in_u):
        in_xu = torch.cat((in_x, in_u), -1)  # concatenate x and u over the last dimension to create the [xu] input
        dx = self.net(in_xu)  # \dot x = f([xu])
        dx = dx * self.scale_dx
        return dx


class ForwardEuler(nn.Module):  # original

    def __init__(self, model, dt=1.0):
        super(ForwardEuler, self).__init__()
        self.model = model
        self.dt = dt

    def forward(self, x0: torch.Tensor, u: torch.Tensor, ahead_step=0) -> torch.Tensor:
        xhat_list = list()

        x_step = x0
        if ahead_step == 1 or ahead_step == 0:
            for u_step in u.split(1):
                u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                xhat_list += [x_step]

            xhat = torch.stack(xhat_list, 0)

            return xhat
        else:
            xhat_ahead = list()
            N = len(u)
            for i in range(N):
                u_step = u[i]
                dx = self.model(x_step, u_step)
                x_step = x_step + dx * self.dt
                xhat_list += [x_step]
                x_ahead = x_step
                if i < N - ahead_step:
                    for j in range(ahead_step):
                        u_step_ahead = u[i + j]
                        dx = self.model(x_ahead, u_step_ahead)
                        x_ahead = x_ahead + dx * self.dt
                    xhat_ahead += [x_ahead]

            xhat = torch.stack(xhat_list, 0)
            xhat_ahead = torch.stack(xhat_ahead, 0)
            return xhat, xhat_ahead





def get_batch(x_fit, U, Y_sys, batch_num, batch_length):
    N = len(Y_sys)
    batch_start = np.random.choice(np.arange(N - batch_length, dtype=np.int64), batch_num, replace=False)
    batch_index = batch_start[:, np.newaxis] + np.arange(batch_length)  # batch sample index
    batch_index = batch_index.T  # (batch_length, batch_num, n_x)
    batch_x0 = x_fit[batch_start, :]  # (batch_num, n_x), initials in each batch
    batch_x = x_fit[[batch_index]]
    batch_u = torch.tensor(U[batch_index, :])
    batch_y = torch.tensor(Y_sys[batch_index])
    return batch_x0, batch_x, batch_u, batch_y


#   --- process to be called ----

def train(data_sample_train, setup):  # train NN

    Y = data_sample_train[0]
    U = data_sample_train[1]
    X = data_sample_train[2]
    dt = setup.dt
    num_epoch = 10000
    simulator = ForwardEuler(model=NeuralStateSpaceModel(n_x=setup.n_x, n_u=setup.n_u,n_feat=setup.hidden, scale_dx= dt), dt=dt)

    N = len(Y)
    Y_sys = np.array(Y, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    Y_sys = Y_sys[:, np.newaxis]
    U = U[:, np.newaxis]

    x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    params_net = list(simulator.model.parameters())
    params_initial = [x_fit]

    optimizer = torch.optim.Adam([
        {'params': params_net, 'lr': setup.lr},
        {'params': params_initial, 'lr': setup.lr}
    ], lr=setup.lr * 10)

    # with torch.no_grad():
    #     batch_x0, batch_x, batch_u, batch_y = get_batch(N, x_fit, U, Y_sys)
    #     batch_xhat = simulator(batch_x0, batch_u)
    #     traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
    #     batch_yhat = batch_xhat[:, :, [0]]
    #     error_init = batch_yhat - batch_y
    #     error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE

    error_scale = torch.tensor([0.01])

    LOSS = []
    for epoch in range(num_epoch):
        batch_x0, batch_x, batch_u, batch_y = get_batch(x_fit, U, Y_sys, setup.batch_num, setup.batch_length)
        batch_xhat = simulator(batch_x0, batch_u)  # traced_

        batch_yhat = batch_xhat[:, :, [0]]
        error_out = batch_yhat - batch_y
        loss_out = torch.mean((error_out / error_scale[0]) ** 2)  # divided by scale
        # state estimate loss
        # error_state = (batch_x - batch_xhat) / error_scale
        # loss_state = torch.mean(error_state ** 2)  # MSE

        loss = loss_out #+ loss_state
        LOSS.append(loss.item())

        if (epoch + 1) % 100 == 0:  # unpack before print
            print(f'epoch {epoch + 1}/{num_epoch}: loss= {loss.item():.5f}, yhat= {batch_yhat[-1, -1, 0]:.4f}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


    params_list = simulator.model.state_dict()

    return [params_list, x_fit]


def test(params, U_test, setup, y=np.ones((2, 1)), ahead_step=0):  # test NN, non-aging
    params_list = params[0]
    x_fit = params[1]
    simulator = ForwardEuler(model=NeuralStateSpaceModel(n_x=setup.n_x, n_u=setup.n_u,n_feat=setup.hidden, scale_dx=setup.dt), dt=setup.dt)

    simulator.model.load_state_dict(params_list, strict=False)  #

    simulator.model.eval()

    x0_vali = x_fit[0, :].detach().numpy()
    x0_vali[1] = 0.0
    x0_vali = torch.tensor(x0_vali.astype(np.float32))
    U = np.array(U_test, dtype=np.float32)
    U = U[:, np.newaxis]
    u_vali = torch.tensor(U)

    with torch.no_grad():

        if ahead_step == 1 or ahead_step == 0:
            xhat_vali = simulator(x0_vali[None, :], u_vali[:, None], ahead_step)
            xhat_vali = xhat_vali.detach().numpy()
            xhat_vali = xhat_vali.squeeze(1)
            yhat_vali = xhat_vali[:, 0]
            return yhat_vali

        else:
            xhat_1, xhat_ahead = simulator(x0_vali[None, :], u_vali[:, None], ahead_step)

            xhat_1 = xhat_1.detach().numpy()
            xhat_1 = xhat_1.squeeze(1)
            yhat_vali = np.array(xhat_1[:, 0])
            xhat_ahead = xhat_ahead.detach().numpy()
            xhat_ahead = xhat_ahead.squeeze(1)
            yhat_ahead = np.array(xhat_ahead[:, 0])
            return yhat_vali, yhat_ahead



# ------- not tested yet >>> ----

def update(Y_sys, U, simulator, params_list, x_fit):  # input model, input pem or others for regulator updating
    N = len(Y_sys)
    checkpoint = params_list
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': online.lr},
    #     {'params': [x_fit], 'lr': online.lr}
    # ], lr=online.lr * 10)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # print(checkpoint.dtype)
    simulator.model.load_state_dict(checkpoint, strict=False)  # , strict=False
    simulator.model.eval()
    # -------------------------------------------------------------------------------------------------------------
    x0 = x_fit[[0], :].detach()
    Y_sys = np.array(Y_sys, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    Y_sys = Y_sys[:, np.newaxis]
    U = U[:, np.newaxis]
    u = torch.tensor(U[:, None, :])
    y = Y_sys[:, np.newaxis]

    xhat_data = simulator(x0, u, y)  # try not given full y_data!!! only partial update
    # ----- optimization inside NN loop, stepwise --------
    yhat = xhat_data[:, 0]
    return yhat

def regulator(params, Y, U, setup, case):  # online regulator
    # Y=data_sample[0]
    # U=data_sample[1]
    params_list = params[0]

    x_fit = params[1]
    N = len(Y)
    threshold1 = 0.91  # start retrain, R2
    threshold2 = 0.95  # stop retrain
    factor = PEM(2, 6, N)
    factor.P_old2 *= 0.09  # 0.009#0.09
    factor.Psi_old2 *= 0.9
    np.random.seed(3)
    factor.Thehat_old = np.random.rand(6, 1) * 0.01
    factor.Xhat_old = np.array([[2], [0]])


    simulator = ForwardEulerPEM(model=MechanicalSystem(setup.dt, setup.n_x, setup.hidden), factor=factor, dt=setup.dt, N=N, update=case, threshold1=threshold1,
                                threshold2=threshold2)
    yhat = update(Y, U, simulator, params_list, x_fit)
    return yhat
