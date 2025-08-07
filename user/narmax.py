import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import sys
import math
from sysidentpy.basis_function import Polynomial
from sysidentpy.model_structure_selection import FROLS
from sysidentpy.parameter_estimation import LeastSquares,RidgeRegression

matplotlib.use("TkAgg")



basis_function = Polynomial(degree=2)
simulator_narmax = FROLS(
    n_terms=4,
    order_selection=False,
    n_info_values=10,
    ylag=2,
    xlag=2,
    info_criteria='aic', 
    estimator=LeastSquares(),  
    basis_function=basis_function
)

#   --- process to be called----
def train(data_sample_train, setup):

    Y = data_sample_train[0]
    U = data_sample_train[1]

    simulator = simulator_narmax
    simulator.fit(X=U.reshape(-1, 1), y=Y.reshape(-1, 1))
    params_list = simulator.theta
    x_fit = Y[0]

    return [params_list, x_fit]


def test(params, U_test, setup,  y=np.ones((2, 1)), ahead_step=0):  # test NN, non-aging
    params_list = params[0]

    simulator = simulator_narmax
    simulator.theta = params_list

    if ahead_step == 0:
       
        yhat = simulator.predict(X=U_test.reshape(-1, 1), y=y.reshape(-1, 1), steps_ahead=1)  # or infinit steps ahead

        return yhat
    else:
        y = y.reshape(-1, 1)
        yhat = simulator.predict(X=U_test.reshape(-1, 1), y=y, steps_ahead=1)  # update X every steps_ahead
        if not isinstance(ahead_step, int):
            ahead_step= ahead_step.item()
        yhat_ahead = simulator.predict(X=U_test.reshape(-1, 1), y=y, steps_ahead=ahead_step)
        return yhat, yhat_ahead




