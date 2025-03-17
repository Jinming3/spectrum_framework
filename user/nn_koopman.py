import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import math
matplotlib.use("TkAgg")


import torch
import torch.nn as nn

# np.random.seed(3)
# torch.manual_seed(3407)
# np.random.seed(0)
# torch.manual_seed(0)


np.random.seed(2681073600) # R2= 0.95 , hidden = 64
torch.manual_seed(1747224464971300)
#
# np.random.seed(2285816605) # R2= 0.94
# torch.manual_seed(1747591965946400)
# seed_n = 0
# seed_t = 0
# np.random.seed(seed_n)
# torch.manual_seed(seed_t)
batch_num = 64
batch_length = 32
n_x = 2

hidden =  128 #64#

print('np seed begin', np.random.get_state()[1][0])
print('torch seed begin ',  torch.seed())

class MechanicalSystem(nn.Module):

    def __init__(self, dt, n_x, hidden, init_small=True):
        super(MechanicalSystem, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = hidden
        self.phi_k = nn.Sequential(nn.Linear(n_x, 2*self.hidden), nn.LeakyReLU(negative_slope=3), nn.Linear(2*self.hidden, self.hidden),# 0.7
                                   nn. ReLU())  #
                                   # nn.ReLU())  # nonlinear to lift x
        # self.phi_b = nn.Linear(1, self.hidden)  # linear to lift u
        self.phi_b = nn.Sequential(nn.Linear(1, 2*self.hidden),nn.Linear(2*self.hidden, self.hidden))#,nn.ReLU(), nn.Linear(128, 64))#,nn.ReLU(),

                                   # nn.ReLU())nn.ReLU(),,nn.ReLU()

        self.inv_phi = nn.Linear(self.hidden, 1)
        # np random seed = train seed
        if init_small:
            for i in self.phi_k.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
            for i in self.phi_b.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)
            for i in self.inv_phi.modules():
                if isinstance(i, nn.Linear):
                    nn.init.normal_(i.weight, mean=0, std=1e-3)
                    nn.init.constant_(i.bias, val=0)



    def forward(self, x1, u1):
        list_dx: List[torch.Tensor]
        self.out_k = self.phi_k(x1)
        self.out_b = self.phi_b(u1)

        self.out_q = self.out_k + self.out_b
        out_inv = self.inv_phi(self.out_q)
        dv = out_inv / self.dt  # v, dv = net(x, v)

        # dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
        list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
        dx = torch.cat(list_dx, -1)
        return dx


class ForwardEuler(nn.Module):  # original

    def __init__(self, model, dt):
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


def vel(pos, dt):
    v_est = np.concatenate((np.array([0]), np.diff(pos[:, 0])))
    v_est = v_est.reshape(-1, 1) / dt
    return v_est


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
    # np.random.seed(seed_n)
    # torch.manual_seed(seed_t)
    print('np seed train', np.random.get_state()[1][0])
    print('torch seed train ', torch.seed())
    Y = data_sample_train[0]
    U = data_sample_train[1]
    dt =  setup.dt
    num_epoch = 21000 #10000
    lr = 0.001 #0.0001
    simulator = ForwardEuler(model=MechanicalSystem(dt,  n_x,  hidden), dt=dt)

    N = len(Y)
    Y_sys = np.array(Y, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    Y_sys = Y_sys[:, np.newaxis]
    U = U[:, np.newaxis]
    v_est = vel(Y_sys, dt)


    X = np.zeros((N,  n_x), dtype=np.float32)
    X[:, 0] = np.copy(Y_sys[:, 0])
    X[:, 1] = np.copy(v_est[:, 0])
    x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    params_net = list(simulator.model.parameters())
    params_initial = [x_fit]

    optimizer = torch.optim.Adam([
        {'params': params_net, 'lr': lr},
        {'params': params_initial, 'lr':  lr}
    ], lr= lr * 10)

    with torch.no_grad():
        batch_x0, batch_x, batch_u, batch_y = get_batch(x_fit, U, Y_sys, batch_num,  batch_length)
        batch_xhat = simulator(batch_x0, batch_u)
        # traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
        batch_yhat = batch_xhat[:, :, [0]]
        error_init = batch_yhat - batch_y
        error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE

    # error_scale = torch.tensor([0.1])

    LOSS = []
    for epoch in range(num_epoch):
        batch_x0, batch_x, batch_u, batch_y = get_batch(x_fit, U, Y_sys,  batch_num,  batch_length)
        batch_xhat = simulator(batch_x0, batch_u)  # traced_

        batch_yhat = batch_xhat[:, :, [0]]
        error_out = batch_yhat - batch_y
        loss_out = torch.mean((error_out / error_scale[0]) ** 2)  # divided by scale
        # state estimate loss
        error_state = (batch_x - batch_xhat) / error_scale
        loss_state = torch.mean(error_state ** 2)  # MSE

        loss = loss_out + loss_state
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
    simulator = ForwardEuler(model=MechanicalSystem(setup.dt,  n_x,  hidden), dt=setup.dt)

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




