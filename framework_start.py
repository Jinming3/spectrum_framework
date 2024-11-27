"""
framework main file, start here
"""
import time
import os
import sys
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from framework.framework_process import sys_select, param_save, param_load, process

matplotlib.use("TkAgg")
# ----- user import >>> ------
sys.path.append('user/')
from user import init

# ------------------ choose a system to generate data
sys_name = 'EMPS'
# sys_name = 'rlc'
# sys_name = 'springs'
# sys_name = 'tanks'


# ---------------------------choose your method to test
# module = 'nn'
# module = 'nn_overflow' # NN for tanks overflow
# module = 'nn_ss' # for non-mechanical like RLC, neural network state-space model
module = 'ss'  # state space
# module = 'narmax'

# module = 'regulator'
# module = 'regulator_overflow'  # for tanks overflow
# module = 'regulator_nn_ss' # for non-mechanical like RLC

# -------------------------- choose to train the model from start or test from loading
new_train = False
# new_train = True

setup = init.ModelTrain_setting(sys_name) # setup hyperparameters

norm = 1 # normalize data into [-norm, +norm]

# # ------------ training ---------
if new_train:
    sample = sys_select(sys_name, setup)
    data_sample_train = sample.sample(time_all=setup.time_train, norm=norm, noise=0)  # data pair for training, [Y, U]

    exec(f'from {module} import train')
    user_params = train(data_sample_train, setup)

    param_save(user_params, sys_name, module)
# # -------  load for trained module----
if not new_train:
    user_params = param_load(sys_name, module)

simulation = process(plot=True)  # to evaluate your model
sample = sys_select(sys_name, setup)
data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm,
                                            noise=10e-5)  # data for testing, only contains data_U
exec(f'from {module} import test')
user_model = test  # using testing dataset
simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='test')


# system under sudden impacts
sample = sys_select(sys_name, setup)
data_sample_change = sample.sample_change(change=setup.change, time_all=setup.time_test, norm=norm) # only U
simulation.test(user_model, user_params, data_sample_change, setup, measure=sample, condition='dynamic')  #


# exec(f'from {module} import regulator')
# user_model = regulator
# simulation.test(user_model, user_params, data_sample_change, setup,measure=sample, condition='dynamic')

# # # # # # test for n-step-ahead prediction, done for SSM --
# sample = sys_select(sys_name,setup)
# data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm, noise=10e-4) #  # data for testing, only contains data_U
# ahead_step= 10#30#5#100
# user_model = test
# simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='ahead', ahead_step=ahead_step)
#  ------------here finishes n-step-ahead #----------

# # # # ---- varying ahead step test ------
# ahead_step_range = [5, 10, 20, 30, 40, 50, 100]
#
# predict_ahead_error =[]
# simulation = process(plot=False)  # to evaluate the user model
#
# for ahead_step in ahead_step_range:
#     sample = sys_select(sys_name, setup)
#     data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm,
#                                                 noise=10e-4)  # # data for testing, only contains data_U
#     exec(f'from {module} import test')
#     user_model = test
#
#     simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='ahead',
#                          ahead_step=ahead_step)
#     predict_ahead_error.append([simulation.RMSE_ahead, simulation.r2_ahead, simulation.fit_ahead])
# predict_ahead_error = np.array(predict_ahead_error, dtype=np.float32)
#
# # np.savetxt(f'ahead_error_{sys_name}_{module}.txt', predict_ahead_error)
#
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(ahead_step_range, predict_ahead_error[:, 0], 'kx-', label='rmse')
# ax[0].legend()

# ax[1].plot(ahead_step_range,predict_ahead_error[:, 1],'bx-', label='r2')
# ax[1].legend()

# # # -----------plot here--------------------
# ahead_step_range = [5, 10, 20, 30, 40, 50, 100]
# error_tanks_tanks = np.loadtxt('ahead_error_tanks_tanks.txt')
# error_tanks_ss = np.loadtxt('ahead_error_tanks_state_space.txt')
# error_tanks_narmax = np.loadtxt('ahead_error_tanks_narmax.txt')
#
# error_emps_nn = np.loadtxt('ahead_error_EMPS_NN.txt')
# error_emps_ss = np.loadtxt('ahead_error_EMPS_state_space.txt')
# error_emps_narmax = np.loadtxt('ahead_error_EMPS_narmax.txt')
#
# error_springs_nn = np.loadtxt('ahead_error_springs_NN.txt')
# error_springs_ss = np.loadtxt('ahead_error_springs_state_space.txt')
# error_springs_narmax = np.loadtxt('ahead_error_springs_narmax.txt')
#
# # --- plot all ---
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(ahead_step_range, error_emps_nn[:, 0], 'rx-', label='nn')
# ax[0].plot(ahead_step_range, error_emps_ss[:, 0], 'bx-', label='ss')
# ax[0].plot(ahead_step_range, error_emps_narmax[:, 0], 'yx-', label='narmax')
# ax[0].legend()
# ax[0].set_xlabel('EMPS')
# ax[1].plot(ahead_step_range, error_springs_nn[:, 0], 'rx-', label='nn')
# ax[1].plot(ahead_step_range, error_springs_ss[:, 0], 'bx-', label='ss')
# ax[1].plot(ahead_step_range, error_springs_narmax[:, 0], 'yx-', label='narmax')
# ax[1].legend()
# ax[1].set_xlabel('springs')
# ax[2].plot(ahead_step_range, error_tanks_tanks[:, 0], 'rx-', label='nn')
# ax[2].plot(ahead_step_range, error_tanks_ss[:, 0], 'bx-', label='ss')
# ax[2].plot(ahead_step_range, error_tanks_narmax[:, 0], 'yx-', label='narmax')
# ax[2].legend()
# ax[2].set_xlabel('tanks')
# fig.suptitle('rmse')
#
# fig, ax = plt.subplots(3, 1, sharex=True)
# ax[0].plot(ahead_step_range, error_emps_nn[:, 1], 'rx-', label='nn')
# ax[0].plot(ahead_step_range, error_emps_ss[:, 1], 'bx-', label='ss')
# ax[0].plot(ahead_step_range, error_emps_narmax[:, 1], 'yx-', label='narmax')
# ax[0].legend()
# ax[0].set_xlabel('EMPS')
# ax[1].plot(ahead_step_range, error_springs_nn[:, 1], 'rx-', label='nn')
# ax[1].plot(ahead_step_range, error_springs_ss[:, 1], 'bx-', label='ss')
# ax[1].plot(ahead_step_range, error_springs_narmax[:, 1], 'yx-', label='narmax')
# ax[1].legend()
# ax[1].set_xlabel('springs')
# ax[2].plot(ahead_step_range, error_tanks_tanks[:, 1], 'rx-', label='nn')
# ax[2].plot(ahead_step_range, error_tanks_ss[:, 1], 'bx-', label='ss')
# ax[2].plot(ahead_step_range, error_tanks_narmax[:, 1], 'yx-', label='narmax')
# ax[2].legend()
# ax[2].set_xlabel('tanks')
# fig.suptitle('r2')








