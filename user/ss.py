import matplotlib.pyplot as plt
import numpy as np


def ss_model(A, B, C, K, U, xhat0, y, ahead_step):
    """
    state space model, no updating when K part not used
    :param A:
    :param B:
    :param C:
    :param u: one u, loop outside ss_model
    :param xhat0: input a xhat [0]
    :return:
    """
    N = len(U)

    yhat_data = []
    y_ahead_data = []
    xhat = xhat0

    for i in range(N):

        yhat = np.dot(C, xhat)
        xhat_new = np.dot(A, xhat) + B * U[i] + K * (y[i] - yhat)

        if ahead_step > 1:

            if i < N - ahead_step:

                x = xhat
                for j in range(ahead_step):
                    x_n = np.dot(A, x) + B * U[i + j]
                    y_n = np.dot(C, x)
                    x = x_n
                y_ahead_data.append(y_n)

        xhat = xhat_new
        yhat_data.append(yhat)

    yhat_data = np.array(yhat_data, dtype=np.float32)
    yhat_data = yhat_data.squeeze(1)

    if ahead_step > 1:
        y_ahead_data = np.array(y_ahead_data, dtype=np.float32)
        y_ahead_data = y_ahead_data.squeeze(1)
        return yhat_data, y_ahead_data
    else:
        return yhat_data


def ss_learn(Thehat_old, P_old2, Psi_old2, Xhat_old, y_all, u_all, n, r, m, t, ahead_step=0):
    """
    train the structural parameters ABCK, x0
    n = dim(A), m = dim(Y), r = dim(U)
    :return:
    """

    N = len(y_all)

    I = np.eye(1)
    Xhatdot0 = np.zeros((n, t))
    Xhatdot_old = np.zeros((n, t))
    Thehat = np.zeros((t, 1))
    Ahat = np.eye(n, n, 1)  # canonical form
    Ahat_old = np.eye(n, n, 1)
    Bhat = np.zeros((n, r))
    Chat = np.eye(m, n)  # [1, 0]  fixed!
    Chat_old = np.eye(m, n)
    Khat = np.zeros((n, m))
    Khat_old = np.zeros((n, m))
    Y_old = np.zeros((m, 1))
    Yhat = np.zeros((m, 1))
    Yhat_old = np.zeros((m, 1))
    U_old = np.zeros((r, 1))
    Xhat = np.zeros((n, 1))
    Yhat_data = np.zeros((N, m))  # collect prediction
    # assign theta-hat
    for a in range(n):
        Ahat[n - 1, a] = Thehat[a, 0]
        Ahat_old[n - 1, a] = Thehat_old[a, 0]
    for b in range(n):
        for b0 in range(r):
            Bhat[b, b0] = Thehat[n + b0 + b * r, 0]
    for h in range(n):  # m=1
        Khat[h, 0] = Thehat[n + n * r + h, 0]
        Khat_old[h, 0] = Thehat_old[n + n * r + h, 0]

    for i in range(N):
        Y = y_all[i]
        U = u_all[[i]]

        for i0 in range(n):  # derivative of A
            Xhatdot0[n - 1, i0] = Xhat_old[i0, 0]
        for i1 in range(n):  # of B
            for i10 in range(r):
                Xhatdot0[i1, n + i1 * r + i10] = U_old[i10]
        for i2 in range(n):  # of K
            Xhatdot0[i2, n + n * r + i2] = Y_old - Yhat_old

        Xhatdot = Xhatdot0 + np.dot(Ahat_old, Xhatdot_old) - np.dot(Khat_old[:, [0]], Psi_old2.T)
        Psi_old = np.dot(Chat_old, Xhatdot).T
        J = I + np.dot(np.dot(Psi_old.T, P_old2), Psi_old)

        P_old = P_old2 - np.dot(np.dot(np.dot(P_old2, Psi_old), np.linalg.pinv(J)),
                                np.dot(Psi_old.T, P_old2))

        Thehat = Thehat_old + np.dot(np.dot(P_old, Psi_old), (Y - Yhat))
        # update thehat
        for a in range(n):
            Ahat[n - 1, a] = Thehat[a, 0]
        for b in range(n):
            for b0 in range(r):
                Bhat[b, b0] = Thehat[n + b0 + b * r, 0]
        for h in range(n):
            Khat[h, 0] = Thehat[n + n * r + h, 0]

        if len(U.shape) == 1:
            U = U.reshape(-1, 1)

        Yhat = np.dot(Chat, Xhat)
        Xhat_new = np.dot(Ahat, Xhat) + Bhat * U + Khat * (Y - Yhat)
        
        # update every parameter which is time-variant
        Xhat_old = np.copy(Xhat)
        Xhat = np.copy(Xhat_new)
        Ahat_old = np.copy(Ahat)
        Khat_old = np.copy(Khat)
        Xhatdot_old = np.copy(Xhatdot)
        Psi_old2 = np.copy(Psi_old)
        U_old = np.copy(U)
        Thehat_old = np.copy(Thehat)
        P_old2 = np.copy(P_old)
        Y_old = np.copy(Y)
        Yhat_old = np.copy(Yhat)
        
        Yhat_data[i] = Yhat
    if ahead_step==0:
        return [Ahat, Bhat, Chat, Khat], Xhat
    elif ahead_step==1:
        return Yhat_data


def train(data_sample_train, setup):
    Y = data_sample_train[0]
    U = data_sample_train[1]

    n = 2
    m = 1
    r = 1
    t = n + n * r + m * n

    np.random.seed(3)
    Thehat_old = np.random.rand(t, 1) * 0.01
    P_old2 = np.eye(t, t) * 0.09
    Psi_old2 = np.eye(t, 1) * 0.9

    Xhat_old = np.random.rand(2, 1)

    if Y.shape[0] == U.shape[0]: # train with full length of Y

        params_list, x_fit = ss_learn(Thehat_old, P_old2, Psi_old2, Xhat_old, Y, U, n, r, m, t)

        return [params_list, x_fit]

def test(params, U_test, setup, y=np.ones((2, 1)), ahead_step=0):  # test model, non-aging
    params_list = params[0]
    x_fit = params[1]
    A = params_list[0]
    B = params_list[1]
    C = params_list[2]
    K = params_list[3]
    xhat = x_fit

    N = len(U_test)

    y = y.reshape(-1, 1)
    U_test = np.array(U_test, dtype=np.float32)
    U = U_test.reshape(-1, 1)

    if len(y) <= 3:
        y = np.zeros((N, 1))

    if ahead_step > 1:  # multiple step ahead
        simulator = ss_model
        yhat, yhat_ahead = simulator(A, B, C, K, U, xhat, y, ahead_step)

        return yhat, yhat_ahead

    elif ahead_step == 1:

        n = 2
        m = 1
        r = 1  #input dimension
        t = n + n * r + m * n

        np.random.seed(3)
        Thehat_old = np.random.rand(t, 1) * 0.01
        P_old2 = np.eye(t, t) * 0.09
        Psi_old2 = np.eye(t, 1) * 0.9
        Xhat_old = np.random.rand(2, 1)
        yhat = ss_learn(Thehat_old, P_old2, Psi_old2, Xhat_old, y, U, n, r, m, t, ahead_step=ahead_step)
        return yhat

    else:  # ahead=0, full prediction without y
        simulator = ss_model
        yhat = simulator(A, B, C, K, U, xhat, y, ahead_step=ahead_step)
        return yhat


