"""
works
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import time
import pykoopman as pk
from pykoopman.common.examples import forced_duffing, rk4, sine_wave  # required for example system
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import numpy.random as rnd
np.random.seed(42)  # for reproducibility

import warnings
warnings.filterwarnings('ignore')


def train(data_sample_train, setup):
    Y0 = data_sample_train[0]
    U = data_sample_train[1]

    n_int = len(Y0)

    Y0 = Y0[:, np.newaxis]
    U = U[:, np.newaxis].T
    dT = setup.ts

    v_est = np.diff(Y0, axis=0) / dT
    v_est = np.r_[[[0]], v_est]  ,
    X = np.concatenate((Y0, v_est), axis=1).T  

    Y = np.roll(X, -1)

    np.random.seed(42)
    EDMDc = pk.regression.EDMDc()
    np.random.seed(42)
    centers = np.random.uniform(-1.5,1.5,(2,4))
    RBF = pk.observables.RadialBasisFunction(
        rbf_type="thinplate",
        n_centers=centers.shape[1],
        centers=centers,
        kernel_width=1,
        polyharmonic_coeff=1.0,
        include_state=True,
    )

    model = pk.Koopman(observables=RBF, regressor=EDMDc)

    off = len(Y0)
    window = 4000
    x_hat_online = np.zeros((off, 2))
    for i in range(off):  
        if i < window:
            Yp = Y[:, :i + 1]
            Up = U[:, :i + 1]
            Xp = X[:, :i + 1]
        else:
            Yp = Y[:, i - window:i]
            Up = U[:, i - window:i]
            Xp = X[:, i - window:i]
        model.fit(Xp.T, y=Yp.T, u=Up.T)

    return model



def test(params, U_test, setup, y=np.ones((2, 1)), ahead_step=0):

    Y0 = y[:, np.newaxis]
    U = U_test[:, np.newaxis].T
    dT = setup.ts
    v_est = np.diff(Y0, axis=0) / dT
    v_est = np.r_[[[0]], v_est]
    X = np.concatenate((Y0, v_est), axis=1).T  # , U
    model= params
    N = len(Y0)
    x_hat_online = np.zeros((N, 2))
    for i in range(N):
        Up = U[:, :i+1]
        x = X[:, i]
        Xkoop_p = model.simulate(x, Up.T)  # window-1, n_steps=2
        x_hat_online[i, :] = Xkoop_p  ##:i+2

    if ahead_step == 1 or ahead_step == 0:
        return x_hat_online[:, 0]
    else:
        x_hat_ahead = np.zeros((N-ahead_step, 2))
        for i in range(N):
            if i < N - ahead_step:
                Up = U[:, :i + ahead_step]
                x = X[:, i]
                Xkoop_p = model.simulate(x, Up.T, n_steps=ahead_step)  # window-1, n_steps=2
                x_hat_ahead[i, :] = Xkoop_p[-1, :]  ##:i+2
        return x_hat_online[:, 0], x_hat_ahead[:, 0]





