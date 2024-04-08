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
from pem import PEM
import torch
import torch.nn as nn


# -------- all trianing settings are here-------
class ModelTrain:
    """
    the only place for users to change these training parameters universally, unless redefined calling a function
    """

    def __init__(self, dt=0.005, change1=20, change2=30, time_train=20, time_test=50):  # 0.005  10  0.3
        self.num_epoch = 10000  # 20000#10000
        self.batch_num = 64
        self.batch_length = 32
        self.lr = 0.0001
        self.n_x = 2
        self.hidden = 64
        self.dt = torch.tensor(dt, dtype=torch.float32)
        self.change1 = change1
        self.change2 = change2
        self.time_train = time_train
        self.time_test = time_test


ModelTrain = ModelTrain()


#  --- >>> original ------
class MechanicalSystem(nn.Module):

    def __init__(self, dt, n_x=ModelTrain.n_x, init_small=True):
        super(MechanicalSystem, self).__init__()
        self.dt = dt  # sampling time
        self.hidden = ModelTrain.hidden
        self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden, bias=True),  # 3*1
                                 # nn.LeakyReLU(negative_slope=0.4),
                                 nn.ReLU(),
                                 nn.Linear(self.hidden, 1, bias=True))  #

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


def vel(pos):
    v_est = np.concatenate((np.array([0]), np.diff(pos[:, 0])))
    v_est = v_est.reshape(-1, 1) / ModelTrain.dt
    return v_est


def get_batch(x_fit, U, Y_sys, batch_num=ModelTrain.batch_num, batch_length=ModelTrain.batch_length):
    N = len(Y_sys)
    batch_start = np.random.choice(np.arange(N - batch_length, dtype=np.int64), batch_num, replace=False)
    batch_index = batch_start[:, np.newaxis] + np.arange(batch_length)  # batch sample index
    batch_index = batch_index.T  # (batch_length, batch_num, n_x)
    batch_x0 = x_fit[batch_start, :]  # (batch_num, n_x), initials in each batch
    batch_x = x_fit[[batch_index]]
    batch_u = torch.tensor(U[batch_index, :])
    batch_y = torch.tensor(Y_sys[batch_index])
    return batch_x0, batch_x, batch_u, batch_y


def train_offline(Y, U, simulator, num_epoch=ModelTrain.num_epoch):  # original
    N = len(Y)
    Y_sys = np.array(Y, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    Y_sys = Y_sys[:, np.newaxis]
    U = U[:, np.newaxis]
    v_est = vel(Y_sys)

    np.random.seed(3)
    torch.manual_seed(3407)
    X = np.zeros((N, ModelTrain.n_x), dtype=np.float32)
    X[:, 0] = np.copy(Y_sys[:, 0])
    X[:, 1] = np.copy(v_est[:, 0])
    x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    params_net = list(simulator.model.parameters())
    params_initial = [x_fit]

    optimizer = torch.optim.Adam([
        {'params': params_net, 'lr': ModelTrain.lr},
        {'params': params_initial, 'lr': ModelTrain.lr}
    ], lr=ModelTrain.lr * 10)

    # with torch.no_grad():
    #     batch_x0, batch_x, batch_u, batch_y = get_batch(N, x_fit, U, Y_sys)
    #     batch_xhat = simulator(batch_x0, batch_u)
    #     traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
    #     batch_yhat = batch_xhat[:, :, [0]]
    #     error_init = batch_yhat - batch_y
    #     error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE
    error_scale = torch.tensor([0.1])

    LOSS = []
    for epoch in range(num_epoch):
        batch_x0, batch_x, batch_u, batch_y = get_batch(x_fit, U, Y_sys)
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

    # # # plot traing result
    # x0_vali = x_fit[0, :].detach().numpy()
    # x0_vali[1] = 0.0
    # x0_vali = torch.tensor(x0_vali)
    # u_vali = torch.tensor(U)
    # with torch.no_grad():
    #     xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
    #     xhat_vali = xhat_vali.detach().numpy()
    #     xhat_vali = xhat_vali.squeeze(1)
    #     yhat_vali = xhat_vali[:, 0]
    #
    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].plot(Y_sys, 'g', label='y')
    # ax[0].plot(yhat_vali, 'r--', label='$\hat{y}$')
    # ax[0].legend()
    #
    # ax[1].plot(U, 'k', label='u')
    # ax[1].set_xlabel('Time')
    # ax[1].legend()
    # # # -----

    params_list = simulator.model.state_dict()

    return params_list, x_fit


def inference(U, simulator, params_list, x_fit):  # input simulator

    simulator.model.load_state_dict(params_list, strict=False)  #

    simulator.model.eval()

    x0_vali = x_fit[0, :].detach().numpy()
    x0_vali[1] = 0.0
    x0_vali = torch.tensor(x0_vali.astype(np.float32))
    U = np.array(U, dtype=np.float32)
    U = U[:, np.newaxis]
    u_vali = torch.tensor(U)

    with torch.no_grad():
        xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
        xhat_vali = xhat_vali.detach().numpy()
        xhat_vali = xhat_vali.squeeze(1)
        yhat_vali = xhat_vali[:, 0]
    return yhat_vali


def update(Y_sys, U, simulator, params_list, x_fit):  # input model, input pem or others for updating
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
    # threshold1 = 0.89  # start retrain, R2
    # threshold2 = 0.90  # stop retrain
    # # threshold2 = 0.97  # stop retrain
    # factor = PEM(2, 6, N)
    # factor.P_old2 *= 0.09
    # factor.Psi_old2 *= 0.9
    # np.random.seed(3)
    # factor.Thehat_old = np.random.rand(6, 1) * 0.1
    # factor.Xhat_old = np.array([[2], [0]])
    # update = 7 #5  # original update
    #
    # simulator = ForwardEulerPEM(model=model, factor=factor, dt=online.dt, N=N, update=update,
    #                             threshold1=threshold1, threshold2=threshold2, step=1000)

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


#   --- process ----
dt = ModelTrain.dt
num_epoch = ModelTrain.num_epoch
model = MechanicalSystem(dt)
simulator = ForwardEuler(model=model, dt=dt)


def train(data_sample_train):  # train NN

    Y = data_sample_train[0]
    U = data_sample_train[1]
    params_list, x_fit = train_offline(Y, U, simulator, num_epoch=num_epoch)

    return [params_list, x_fit]


def test(params, U_test, simulator=simulator):  # test NN, non-aging
    params_list = params[0]
    x_fit = params[1]
    yhat = inference(U_test, simulator, params_list, x_fit)
    return yhat


def regulator(params, Y, U, case):  # train dyn
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

    case = case
    simulator = ForwardEulerPEM(model=model, factor=factor, dt=dt, N=N, update=case,threshold1=threshold1, threshold2=threshold2)

    yhat = update(Y, U, simulator, params_list, x_fit)
    return yhat


#  ------ <<< original, don't delete ---

# class MechanicalSystem_linear(nn.Module):  # linear update weights, not working,deletable
#
#     def __init__(self, dt, n_x=ModelTrain.n_x, init_small=True, LinearParams=torch.ones(4, ModelTrain.hidden), LinearUpdate=False):
#         super(MechanicalSystem_linear, self).__init__()
#         self.dt = dt  # sampling time
#         self.hidden = ModelTrain.hidden
#         self.net = nn.Sequential(nn.Linear(n_x + 1, self.hidden, bias =True),  # 3*1
#                                  # nn.LeakyReLU(negative_slope=0.4),
#                                  nn.ReLU(),
#                                  nn.Linear(self.hidden, 1, bias=True))  #
#         self.LinearParams = LinearParams
#         self.LinearUpdate=LinearUpdate
#         self.frozen1= torch.tensor([1])
#         self.frozen2= torch.tensor([1])#.data.copy_(self.net[2].weight)
#
#
#         if init_small:
#             for i in self.net.modules():
#                 if isinstance(i, nn.Linear):
#                     nn.init.normal_(i.weight, mean=0, std=1e-3)
#                     nn.init.constant_(i.bias, val=0)
#
#     def forward(self, x1, u1):   #, fix1, fix2
#         if self.LinearUpdate==True:
#             self.net[0].weight.data.copy_(self.frozen1 * self.LinearParams[0:3, :].T)  #
#             self.net[2].weight.data.copy_(self.frozen2 * self.LinearParams[3, :])
#         list_dx: List[torch.Tensor]
#         in_xu = torch.cat((x1, u1), -1)
#         dv = self.net(in_xu) / self.dt  # v, dv = net(x, v)
#         list_dx = [x1[..., [1]], dv]  # [dot x=v, dot v = a]
#         dx = torch.cat(list_dx, -1)
#         return dx


def train_offline_projection(Y, U, simulator, LinearParams, LinearUpdate=False,
                             num_epoch=ModelTrain.num_epoch):  # for update C
    N = len(Y)
    Y_sys = np.array(Y, dtype=np.float32)
    U = np.array(U, dtype=np.float32)
    Y_sys = Y_sys[:, np.newaxis]
    U = U[:, np.newaxis]
    v_est = vel(Y_sys)

    np.random.seed(3)
    torch.manual_seed(3407)
    X = np.zeros((N, ModelTrain.n_x), dtype=np.float32)
    X[:, 0] = np.copy(Y_sys[:, 0])
    X[:, 1] = np.copy(v_est[:, 0])
    x_fit = torch.tensor(X, dtype=torch.float32, requires_grad=True)

    if LinearUpdate == False:
        params_net = list(simulator.model.parameters())
        params_initial = [x_fit]

        optimizer = torch.optim.Adam([
            {'params': params_net, 'lr': ModelTrain.lr},
            {'params': params_initial, 'lr': ModelTrain.lr}
        ], lr=ModelTrain.lr * 10)
    if LinearUpdate == True:
        # params = LinearParams
        params = list(LinearParams.parameters())
        optimizer = torch.optim.Adam([
            {'params': params, 'lr': ModelTrain.lr}
        ], lr=ModelTrain.lr * 10)

    # with torch.no_grad():
    #     batch_x0, batch_x, batch_u, batch_y = get_batch(N, x_fit, U, Y_sys)
    #     batch_xhat = simulator(batch_x0, batch_u)
    #     traced_simulator = torch.jit.trace(simulator, (batch_x0, batch_u))
    #     batch_yhat = batch_xhat[:, :, [0]]
    #     error_init = batch_yhat - batch_y
    #     error_scale = torch.sqrt(torch.mean(error_init ** 2, dim=(0, 1)))  # root MSE
    error_scale = torch.tensor([0.1])

    LOSS = []
    # LOSS_initial = []
    # LOSS_output = []
    for epoch in range(num_epoch):
        batch_x0, batch_x, batch_u, batch_y = get_batch(N, x_fit, U, Y_sys)
        batch_xhat = simulator(batch_x0, batch_u)  # traced_
        # output loss
        if LinearUpdate == False:
            batch_yhat = batch_xhat[:, :, [0]]
        if LinearUpdate == True:
            batch_yhat = LinearParams(batch_xhat)
        error_out = batch_yhat - batch_y
        loss_out = torch.mean((error_out / error_scale[0]) ** 2)  # divided by scale
        # state estimate loss
        error_state = (batch_x - batch_xhat) / error_scale
        loss_state = torch.mean(error_state ** 2)  # MSE
        # if epoch > 1000:
        #     loss = loss_out + weight*loss_state
        # else:
        #     loss = loss_out
        loss = loss_out + ModelTrain.weight * loss_state
        LOSS.append(loss.item())
        # LOSS_initial.append(loss_state.item())
        # LOSS_output.append(loss_out.item())

        if (epoch + 1) % 100 == 0:  # unpack before print
            print(f'epoch {epoch + 1}/{num_epoch}: loss= {loss.item():.5f}, yhat= {batch_yhat[-1, -1, 0]:.4f}')

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # params_list = simulator.model.state_dict()
    params_list = {'model_state_dict': simulator.model.state_dict()}

    # params_list = {'model_state_dict': simulator.model.state_dict(),
    #               'optimizer_state_dict': optimizer.state_dict()}

    return params_list, x_fit
    # return simulator


def inference_projection(U, simulator, params_list, x_fit, ahead_step, LinearParams, projection):  # input simulator
    checkpoint = params_list
    simulator.model.load_state_dict(checkpoint, strict=False)  #
    # optimizer = torch.optim.Adam([
    #     {'params': model.parameters(), 'lr': online.lr},
    #     {'params': [x_fit], 'lr': online.lr}
    # ], lr=online.lr * 10)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    simulator.model.eval()

    # simulator = ForwardEuler(model=model, dt=online.dt)

    x0_vali = x_fit[0, :].detach().numpy()
    x0_vali[1] = 0.0
    x0_vali = torch.tensor(x0_vali)
    U = np.array(U, dtype=np.float32)
    U = U[:, np.newaxis]
    u_vali = torch.tensor(U)

    with torch.no_grad():
        xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
        if projection == False:
            xhat_vali = xhat_vali.detach().numpy()
            xhat_vali = xhat_vali.squeeze(1)
            yhat_vali = xhat_vali[:, 0]
        if projection == True:
            xhat_vali = xhat_vali.squeeze(1)
            yhat_vali = LinearParams(xhat_vali)
            yhat_vali = yhat_vali.detach().numpy()

    return yhat_vali


def inference_online_projection(U, Y, model, params_list, x_fit, ahead_step, LinearParams,
                                projection):  # update C, wrong!
    checkpoint = params_list
    model.load_state_dict(checkpoint, strict=False)  #

    model.eval()
    x0 = x_fit[0, :].detach().numpy()
    x0[1] = 0.0
    x0 = torch.tensor(x0)
    U = np.array(U, dtype=np.float32)
    U = U[:, np.newaxis]
    u = torch.tensor(U)
    Y = torch.tensor(Y)

    if projection == False:
        xhat_list = list()
        x_step = x0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx = self.model(x_step, u_step)
            x_step = x_step + dx * self.dt
            xhat_list += [x_step]

        xhat = torch.stack(xhat_list, 0)

        # xhat_vali = simulator(x0_vali[None, :], u_vali[:, None])
        xhat = xhat.detach().numpy()
        xhat = xhat.squeeze(1)
        yhat_vali = xhat[:, 0]

    if projection == True:
        lrh = 0.00001
        yhat_list = list()
        params = list(LinearParams.parameters())
        optimizer = torch.optim.Adam([
            {'params': params, 'lr': ModelTrain.lr}  # ModelTrain.lr
        ], lr=ModelTrain.lr)

        error_scale = torch.tensor([0.1])

        x_step = x0
        linear_update = 0
        for u_step in u.split(1):
            u_step = u_step.squeeze(0)  # size (1, batch_num, 1) -> (batch_num, 1)
            dx = model(x_step, u_step)
            x_step = x_step + dx * ModelTrain.dt
            yhat = LinearParams(x_step)
            yhat_list += [yhat]
            error = yhat - Y[linear_update]
            loss = torch.mean((error / error_scale[0]) ** 2)  # divided by scale

            if (linear_update + 1) % 100 == 0:  # unpack before print
                print(f'{linear_update + 1}/{len(U)}: loss= {loss.item():.3f}, yhat= {yhat.item():.4f}')

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            linear_update = linear_update + 1

        yhat_vali = torch.stack(yhat_list, 0)
    yhat_vali = yhat_vali.clone().detach().numpy()
    return yhat_vali
