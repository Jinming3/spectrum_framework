""""
functions called by framework_start, no access to user
"""
import numpy as np
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error
from math import sqrt
import sys, os
sys.path.append('F:/Project/inProcess/Framework/user/')
# os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'user/'))
from user_process import ModelTrain
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


def sinwave(A, dt, time, w=0.5, sig=1, phi=0):
    """
    sin/cos wave signal
    w: frequency
    A: amplitude
    """
    out = []
    for k in range(int(time / dt)):
        if sig == 1:
            x = A * np.cos(w * k * math.pi * dt + phi)
        if sig == 0:
            x = A * np.sin(w * k * math.pi * dt + phi)
        out.append(x)
    return out


def triangle(A, dt, time_all, p=2):
    """
    triangle wave signal
    A: amplitude
    p: period
    """
    out = []
    for k in range(int(time_all / dt)):
        x = A * 2 * np.abs(k * dt / p - math.floor(k * dt / p + 0.5))
        out.append(x)
    return out


def p_ref(A, dt, time_all, signal='sinwave', w=0.5, p=2):
    """
    generate reference signal, sin or tri wave
    A: amplitude
    """
    if signal == 'sinwave':
        out = sinwave(A, dt, time_all, w=w)
    if signal == 'triangle':
        out = triangle(A, dt, time_all, p=p)
    return out


class EMPS:
    def __init__(self, dt, pos=0, vel=0, acc=0, u=0, gt=35.15, kp=160.18, kv=243.45, M=95.1089, Fv=203.5034,
                 Fc=20.3935, offset=-3.1648, satu=10, limit=20):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.u = u  # control signal voltage
        self.dt = dt
        self.gt = gt
        self.kp = kp
        self.kv = kv
        self.M = M
        self.Fv = Fv
        self.Fc = Fc
        self.offset = offset
        self.satu = satu
        self.limit = limit  # position frame limition

    def measure(self, pos_ref, noise_process=0.0, noise_measure=0.0):
        self.u = self.kp * self.kv * (pos_ref - self.pos) - self.kv * self.vel
        if self.u > self.satu:  # Maximum voltage (saturation)
            self.u = self.satu
        if self.u < -self.satu:
            self.u = -self.satu
        force = self.gt * self.u
        self.acc = force / self.M - self.Fv / self.M * self.vel - self.Fc / self.M * np.sign(
            self.vel) - self.offset / self.M + np.random.randn() * noise_process
        self.vel = self.vel + self.acc * self.dt
        self.pos = self.pos + self.vel * self.dt
        # position limit
        # if self.pos > self.limit:
        #     self.pos = self.limit
        # if self.pos < -self.limit:
        #     self.pos = -self.limit
        output = self.pos + np.random.randn() * noise_measure
        return output


class sys_select:
    """
    select a system to generate data
    """

    def __init__(self, sys_name, dt):
        if sys_name == 'EMPS':
            self.system = EMPS(dt=dt)
        self.dt = dt

    def sample(self, A, time_all, signal, norm=0, noise=0):
        """
        original system or noisy system, noise=0.0001 (a scale)
        """
        self.Y = []
        self.U = []

        ref = p_ref(A, self.dt, time_all, signal=signal)

        for p_control in ref:
            y = self.system.measure(p_control, noise * 10, noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        if norm == 0:
            self.Y_hidden = self.Y
            return [self.Y, self.U]

        if norm > 0:
            self.Y = normalize(self.Y, r=norm)
            self.U = normalize(self.U, r=norm)
            self.Y_hidden = Y
            return [self.Y, self.U]

    def sample_infer(self, A, time_all, signal, norm=0, noise=0.00001):
        """
        # original system or noisy system, non-aging
        """

        self.Y = []
        self.U = []

        ref = p_ref(A, self.dt, time_all, signal=signal)

        for p_control in ref:
            y = self.system.measure(p_control, noise * 10, noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y_hidden = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric
        self.U = np.array(self.U, dtype=np.float32)

        if norm == 0:
            return self.U

        if norm > 0:
            self.Y_hidden = normalize(self.Y_hidden, r=norm)
            self.U = normalize(self.U, r=norm)
            return self.U

    def sample_aging(self, A, change1, change2, time_all, signal, aging_rate, norm=0, noise=0.0001):
        """
        # system parameters aging, no reference signal changing
        """

        self.Y = []
        self.U = []

        N = int(time_all / self.dt)
        change1 = int(change1 / self.dt)
        change2 = int(change2 / self.dt)
        ref = p_ref(A, self.dt, time_all, signal=signal)

        # aging starts or comment it out for original
        self.system.offset = self.system.offset * 0.99
        self.system.M = self.system.M * aging_rate
        self.system.Fv = self.system.Fv * aging_rate
        self.system.Fc = self.system.Fc * aging_rate

        for i in range(change1):  # original
            y = self.system.measure(ref[i], noise_process=noise, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)
        # aging starts
        self.system.offset = self.system.offset * 0.99
        self.system.M = self.system.M * aging_rate
        self.system.Fv = self.system.Fv * aging_rate
        self.system.Fc = self.system.Fc * aging_rate
        for i in range(change1, change2):
            y = self.system.measure(ref[i], noise_process=noise*10, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)
        # aging
        self.system.offset = self.system.offset * 0.99
        self.system.M = self.system.M * aging_rate
        self.system.Fv = self.system.Fv * aging_rate
        self.system.Fc = self.system.Fc * aging_rate

        for i in range(change2, N):
            y = self.system.measure(ref[i], noise_process=10 *noise, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y_hidden = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        if norm == 0:
            return self.U

        if norm > 0:
            self.Y_hidden = normalize(self.Y_hidden, r=norm)
            self.U = normalize(self.U, r=norm)
            return self.U

    def sample_ref(self, A, change1, change2, time_all, norm=0, noise=0.0001):
        """
        # no aging, only reference signal changing
        """

        N = int(time_all / self.dt)
        change1 = int(change1 / self.dt)
        change2 = int(change2 / self.dt)
        ref1 = p_ref(A, self.dt, time_all, signal='sinwave')
        ref2 = p_ref(A, self.dt, time_all, signal='triangle')
        ref3 = p_ref(A, self.dt, time_all, signal='sinwave', w=0.8)
        self.ref_signal = []
        self.Y = []
        self.U = []

        for i in range(change1):  # original
            y = self.system.measure(ref1[i], noise_process=noise, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)

            self.ref_signal.append(ref1[i])

        for i in range(change1, change2):  # change to ref2
            y = self.system.measure(ref2[i], noise_process=10 *noise, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)
            self.ref_signal.append(ref2[i])

        for i in range(change2, N):  # change back to ref1
            y = self.system.measure(ref3[i], noise_process=10 *noise, noise_measure=noise)
            self.Y.append(y)
            self.U.append(self.system.u)
            self.ref_signal.append(ref1[i])

        self.Y_hidden = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)
        self.ref_signal = np.array(self.ref_signal, dtype=np.float32)

        if norm == 0:
            return self.U

        if norm > 0:
            self.Y_hidden = normalize(self.Y_hidden, r=norm)
            self.U = normalize(self.U, r=norm)
            return self.U





class process:  # training or test
    def __init__(self, plot):
        self.plot = plot

    def train(self, user_model, data_sample_train):
        params, x_init = user_model(data_sample_train)
        return params, x_init

    def inference(self, user_model, user_params, data_sample_U, measure, condition, case=0):

        if condition == 'dynamic':  # make one step ahead prediction, with r2 threshold
            Y = measure.Y_hidden
            self.yhat = user_model(user_params,  Y, data_sample_U, case=case)


        elif condition == 'user':
            self.yhat = user_model(user_params, data_sample_U)
            Y = measure  # directly input y for plot

        else:
            self.yhat = user_model(user_params, data_sample_U)  # for non-aging test
            Y = measure.Y_hidden


        self.RMSE = rmse(Y, self.yhat)
        print(f'inference_{condition}_RMSE=', self.RMSE)

        self.r2 = R2(Y, self.yhat)
        print(f'inference_{condition}_R2=', self.r2)



        if self.plot:
            N = len(Y)  # length of Y
            time_exp = np.arange(N)*ModelTrain.dt.clone().detach().numpy()
            fig, ax = plt.subplots(2, 1, sharex=True)
            ax[0].plot(time_exp, data_sample_U, 'k', label='u')
            ax[0].set_xlabel('Time')
            ax[0].legend()
            ax[1].plot(time_exp, Y, 'g', label='y')
            ax[1].plot(time_exp, self.yhat, 'r', label=f'yhat_{condition}')
            if condition=='dynamic':
                change1 = ModelTrain.change1
                change2 = ModelTrain.change2
                changing = np.array([change1 / ModelTrain.dt, change2 / ModelTrain.dt]).astype(int)
                ax[1].plot(time_exp[changing], Y[changing], 'kx')
            ax[1].legend()
            # plt.show()




