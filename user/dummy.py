import numpy as np


def model(Y, U,ahead_step=0):
    N = len(U)
    yhat_data = []
    y_ahead_data = []

    y = Y[0]
    for i in range(0,N):
        y_new = Y[i]
        yhat_data.append(y)
        if ahead_step > 1:
            if i < N - ahead_step:
                for j in range(ahead_step):
                    y_n = y
                    y = y_n
                y_ahead_data.append(y_n)

        y = y_new


    yhat_data = np.array(yhat_data)
    if ahead_step > 1:
        y_ahead_data = np.array(y_ahead_data, dtype=np.float32)
        return yhat_data, y_ahead_data
    else:
        return yhat_data


def train(data_sample_train, setup, ahead_step=0):
    Y = data_sample_train[0]
    U = data_sample_train[1]

    yhat_data = model(Y, U)

def test(user_params, data_sample_U, setup, y, ahead_step=0):
    U = data_sample_U
    Y = y
    yhat_data = model(Y, U, ahead_step)
    return yhat_data







