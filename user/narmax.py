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



# basis_function = Polynomial(degree=1)
# simulator_narmax = FROLS(
#     order_selection=True,
#     n_info_values=10,
#     # extended_least_squares=False,
#     ylag=2,
#     xlag=2,
#     info_criteria='aic',
#     # estimator='least_squares',  # 'LeastSquares'
#     basis_function=basis_function
# )
basis_function = Polynomial(degree=2)
simulator_narmax = FROLS(
    n_terms=4,
    order_selection=False,#True, #
    n_info_values=10,
    # extended_least_squares=True,
    ylag=2,
    xlag=2,
    info_criteria='aic', #'lilc',#'bic',#
    estimator=LeastSquares(),  #
    # estimator = RidgeRegression(lam=0.01),
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

    # x_fit = params[1]
    simulator = simulator_narmax
    simulator.theta = params_list

    if ahead_step == 0:
        # yhat = simulator.simulate(X=U_test.reshape(-1, 1), y=y.reshape(-1, 1))  #, steps_ahead=1, infinit steps ahead
        yhat = simulator.predict(X=U_test.reshape(-1, 1), y=y.reshape(-1, 1), steps_ahead=1)  #, infinit steps ahead

        return yhat
    else:
        y = y.reshape(-1, 1)
        yhat = simulator.predict(X=U_test.reshape(-1, 1), y=y, steps_ahead=1)  # update X every steps_ahead
        if not isinstance(ahead_step, int):
            ahead_step= ahead_step.item()
        yhat_ahead = simulator.predict(X=U_test.reshape(-1, 1), y=y, steps_ahead=ahead_step)
        return yhat, yhat_ahead




