""""
functions called by framework_start, no access to user
"""
import math
import os
import sys
from math import sqrt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error

sys.path.append('F:/Project/inProcess/Framework/framework/system/')
# sys.path.append('system/')
# import system
import emps, rlc, tanks, springs

matplotlib.use("TkAgg")


def rmse(Y_sys, Yhat):
    return sqrt(mean_squared_error(Y_sys, Yhat))


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


def fit_index(y_true, y_pred, time_axis=0):
    """ Computes the per-channel fit index.

    The fit index is commonly used in System Identification. See the definitionin the System Identification Toolbox
    or in the paper 'Nonlinear System Identification: A User-Oriented Road Map',
    https://arxiv.org/abs/1902.00683, page 31.
    The fit index is computed separately on each channel.

    Parameters
    ----------
    y_true : np.array
        Array of true values.  If must be at least 2D.
    y_pred : np.array
        Array of predicted values.  If must be compatible with y_true'
    time_axis : int
        Time axis. All other axes define separate channels.

    Returns
    -------
    fit_val : np.array
        Array of r_squared value.

    """

    err_norm = np.linalg.norm(y_true - y_pred, axis=time_axis, ord=2)  # || y - y_pred ||
    y_mean = np.mean(y_true, axis=time_axis)
    err_mean_norm = np.linalg.norm(y_true - y_mean, ord=2)  # || y - y_mean ||
    fit_val = 100*(1 - err_norm/err_mean_norm)

    return fit_val


def normalize(x, r=1):
    """
    normalize an array # 1-dimension
    :param x: the original array
    :param r: new array of [-r, r]
    :return: new array
    """
    out = []
    mini = np.amin(x)
    maxi = np.amax(x)
    for j in range(len(x)):
        # norm = (x[i] - mini) / (maxi - mini)  # [0, 1]
        norm = 2 * r * (x[j] - mini) / (maxi - mini) - r
        out.append(norm)
    out = np.array(out, dtype=np.float32)
    return out


class sys_select:
    """
    select a system to generate data
    """

    def __init__(self, sys_name, setup):
        self.sys_name = sys_name
        if sys_name == 'EMPS':
            self.system = emps.measure(dt=setup.ts)
        if sys_name == 'springs':
            self.system = springs.measure(ts=setup.ts)
        if sys_name == 'tanks':
            self.system = tanks.measure(dt=setup.ts)
        if sys_name == 'rlc':
            self.system = rlc.measure(ts=setup.ts)

    def sample(self, time_all, norm, noise):
        """
        original system or noisy system for training
        """
        self.Y, self.U = self.system.sample(time_all, noise=noise)
        if self.sys_name == 'rlc':
            self.X = self.system.X

        if norm == 0:
            self.Y_hidden = self.Y
            if self.sys_name == 'rlc':
                return [self.Y, self.U, self.X]
            else:
                return [self.Y, self.U]

        if norm > 0:
            self.Y = normalize(self.Y, r=norm)
            self.U = normalize(self.U, r=norm)
            self.Y_hidden = self.Y
            if self.sys_name == 'rlc':
                self.X = normalize(self.X, r=norm)
                return [self.Y, self.U, self.X]
            else:
                return [self.Y, self.U]

    def sample_test(self, time_all, norm, noise):
        """
        # original system with noise, non-aging
        """

        self.Y, self.U = self.system.sample_test(time_all, noise=noise)

        self.Y_hidden = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric

        if norm == 0:
            return self.U

        if norm > 0:
            self.Y_hidden = normalize(self.Y_hidden, r=norm)
            self.U = normalize(self.U, r=norm)
            return self.U

    def sample_change(self, change, time_all, norm):
        """
        # system parameters aging
        """

        self.Y, self.U = self.system.sample_change(change, time_all)
        self.Y_hidden = np.array(self.Y, dtype=np.float32)

        if norm == 0:
            return self.U

        if norm > 0:
            self.Y_hidden = normalize(self.Y_hidden, r=norm)
            self.U = normalize(self.U, r=norm)
            return self.U


def param_save(user_params, sys_name, module):
    model_folder = os.path.join("models", sys_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(user_params, os.path.join(model_folder, f"{sys_name}_{module}"))  # also for non-NN module


def param_load(sys_name, module):
    model_folder = os.path.join("models", sys_name)
    user_params = torch.load(os.path.join(model_folder, f"{sys_name}_{module}"))
    return user_params


class process:  # training or test
    def __init__(self, plot):
        self.plot = plot

    def test(self, user_model, user_params, data_sample_U, setup, measure, condition, ahead_step=0):

        if condition == 'ahead':
            Y = measure.Y_hidden  # predict n step ahead
            if ahead_step == 0 or ahead_step == 1:  # no update or one-step ahead
                self.yhat = user_model(user_params, data_sample_U, setup, y=Y, ahead_step=ahead_step)
            else:  # one and more step ahead, update with y
                self.yhat, self.yhat_ahead = user_model(user_params, data_sample_U, setup, y=Y, ahead_step=ahead_step)

        else:
            Y = measure.Y_hidden  # real Y is not provided for update
            self.yhat = user_model(user_params, data_sample_U, setup, y=Y)  # for non-aging test


        # if len(self.yhat.shape) > 1:
        #     self.yhat = self.yhat[:, 0]
        # self.RMSE = rmse(Y, self.yhat)
        # self.r2 = R2(Y, self.yhat)
        # print(f'test_{condition}_RMSE=', self.RMSE)
        # print(f'test_{condition}_R2=', self.r2)
        #
        # if condition == 'ahead':
        #     if len(Y) != len(self.yhat_ahead):
        #         if len(self.yhat_ahead.shape) == 2:
        #             self.yhat_ahead = self.yhat_ahead[:, 0]
        #
        #         self.RMSE_ahead = rmse(Y[ahead_step:], self.yhat_ahead)
        #         self.r2_ahead = R2(Y[ahead_step:], self.yhat_ahead)
        #
        #         print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
        #         print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)
        #     if len(Y) == len(self.yhat_ahead):
        #         self.RMSE_ahead = rmse(Y, self.yhat_ahead[:, 0])
        #         self.r2_ahead = R2(Y, self.yhat_ahead[:, 0])
        #         print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
        #         print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)

        # # -----
        # if ahead_step == 0 or ahead_step == 1:  # full prediction or one-step-ahead
        #
        #     if len(self.yhat.shape) > 1:  # extract array in single-axis
        #         self.yhat = self.yhat[:, 0]
        #     self.RMSE = rmse(Y, self.yhat)
        #     self.r2 = R2(Y, self.yhat)
        #     print(f'test_{condition}_RMSE=', self.RMSE)
        #     print(f'test_{condition}_R2=', self.r2)
        #
        # else: # contains multiple step ahead
        #     if len(self.yhat.shape) > 1:
        #         self.yhat = self.yhat[:, 0]
        #     self.RMSE = rmse(Y, self.yhat)
        #     self.r2 = R2(Y, self.yhat)
        #     print(f'test_{condition}_RMSE=', self.RMSE)
        #     print(f'test_{condition}_R2=', self.r2)
        #
        #     if len(Y) != len(self.yhat_ahead):
        #         if len(self.yhat_ahead.shape) == 2:
        #             self.yhat_ahead = self.yhat_ahead[:, 0]
        #
        #         self.RMSE_ahead = rmse(Y[ahead_step:], self.yhat_ahead)
        #         self.r2_ahead = R2(Y[ahead_step:], self.yhat_ahead)
        #
        #         print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
        #         print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)
        #
        #     if len(Y) == len(self.yhat_ahead):
        #         self.RMSE_ahead = rmse(Y, self.yhat_ahead[:, 0])
        #         self.r2_ahead = R2(Y, self.yhat_ahead[:, 0])
        #         print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
        #         print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)
        # # -----
        # -----

        # evaluation
        if len(self.yhat.shape) > 1:  # data prepare, extract array in single-axis
            self.yhat = self.yhat[:, 0]
        self.RMSE = rmse(Y, self.yhat)
        self.r2 = R2(Y, self.yhat)
        self.fit = fit_index(Y, self.yhat)
        print(f'test_{condition}_RMSE=', self.RMSE)
        print(f'test_{condition}_R2=', self.r2)
        print(f'test_{condition}_fit(%)=', self.fit)

        if ahead_step>1:  # contains multiple step ahead
            if len(Y) != len(self.yhat_ahead):
                if len(self.yhat_ahead.shape) == 2:  # data prepare, extract array in single-axis
                    self.yhat_ahead = self.yhat_ahead[:, 0]

                self.RMSE_ahead = rmse(Y[ahead_step:], self.yhat_ahead)
                self.r2_ahead = R2(Y[ahead_step:], self.yhat_ahead)
                self.fit_ahead = fit_index(Y[ahead_step:], self.yhat_ahead)

                # print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
                # print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)
                # print(f'predict_ahead_{ahead_step}_fit=', self.fit_ahead)

            if len(Y) == len(self.yhat_ahead):
                self.RMSE_ahead = rmse(Y, self.yhat_ahead[:, 0])
                self.r2_ahead = R2(Y, self.yhat_ahead[:, 0])
                self.fit_ahead = fit_index(Y, self.yhat_ahead[:, 0])

            print(f'predict_ahead_{ahead_step}_RMSE=', self.RMSE_ahead)
            print(f'predict_ahead_{ahead_step}_R2=', self.r2_ahead)
            print(f'predict_ahead_{ahead_step}_fit(%)=', self.fit_ahead)
        # -----

        # if ahead_step>1:
        #
        #     if len(self.yhat.shape) == 1:
        #         if len(self.yhat_ahead.shape)==2:
        #             self.yhat_ahead = self.yhat_ahead[:, 0]
        #         print(f'predict_ahead_{ahead_step}_RMSE=', rmse(Y[ahead_step:], self.yhat_ahead))
        #         print(f'predict_ahead_{ahead_step}_R2=', R2(Y[ahead_step:], self.yhat_ahead))
        #     elif len(Y)==len(self.yhat_ahead):
        #         print(f'predict_ahead_{ahead_step}_RMSE=', rmse(Y, self.yhat_ahead[:, 0]))
        #         print(f'predict_ahead_{ahead_step}_R2=', R2(Y, self.yhat_ahead[:, 0]))
        #     else:
        #         print(f'predict_ahead_{ahead_step}_RMSE=', rmse(Y[ahead_step:], self.yhat_ahead[:, 0]))
        #         print(f'predict_ahead_{ahead_step}_R2=', R2(Y[ahead_step:], self.yhat_ahead[:, 0]))

        if self.plot:
            N = len(Y)  # length of Y
            time_exp = np.arange(N) * setup.ts
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(time_exp, data_sample_U, 'k', label='u')
            ax[0].legend()
            ax[1].set_xlabel('Time')
            ax[1].plot(time_exp, Y, 'g', label='y')
            ax[1].plot(time_exp, self.yhat, 'r', label=f'yhat_{condition}')
            if condition == 'dynamic':  # mark up the changing points
                change = (setup.change / setup.ts).astype(int)
                ax[1].plot(time_exp[change], Y[change], 'kx')
            if condition == 'ahead' and ahead_step > 1 and len(Y) != len(self.yhat_ahead):
                ax[1].plot(time_exp[ahead_step:], self.yhat_ahead, 'b--', label=f'yhat_ahead_{ahead_step}')
            elif condition == 'ahead' and ahead_step > 1 and len(Y) == len(self.yhat_ahead):
                ax[1].plot(time_exp, self.yhat_ahead, 'b--', label=f'yhat_ahead_{ahead_step}')
            ax[1].legend()
