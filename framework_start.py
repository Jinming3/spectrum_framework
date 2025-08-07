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

# ---------- choose a system to generate data -----------
sys_name = 'EMPS'
# sys_name = 'springs'
# sys_name = 'rlc'
# sys_name = 'tanks'

# ------------choose your method to test -----------------
# module = 'dummy'
# module = 'ss'  # state space
# module = 'nn'
module = 'nn_koopman'
# module = 'narmax'

# module = 'nn_overflow' # NN for tanks overflow
# module = 'nn_ss' # for non-mechanical like RLC, neural network state-space model

# ----------- choose to train the model from start or test from loading ---------------
new_train = False
# new_train = True
version = 0
setup = init.ModelTrain_setting(sys_name)  # setup hyperparameters
norm = 1  # normalize data into [-norm, +norm]

# # ------------ training ---------
if new_train:
    sample = sys_select(sys_name, setup)
    data_sample_train = sample.sample(time_all=setup.time_train, norm=norm, noise=0)  # data pair for training, [Y, U]
    np.savetxt('ref_train.txt', sample.system.ref)

    exec(f'from {module} import train')
    start_train_time = time.time()
    user_params = train(data_sample_train, setup)
    print(f"\nTrain_time_{module}: {time.time() - start_train_time:.2f}")
    param_save(user_params, sys_name, module)
# # -------  load for trained module ----
if not new_train:
    user_params = param_load(sys_name, module, version=0)

simulation = process(plot=True)  # to evaluate your model,
sample = sys_select(sys_name, setup)
data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm,
                                      noise=10e-5)  # data for testing, only contains data_U

# np.savetxt('data_test.txt', [sample.system.Y, sample.system.U])

exec(f'from {module} import test')
user_model = test  # using testing dataset

simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='test')
# np.savetxt(f'data_test_yhat_{sys_name}_{module}.txt', simulation.yhat)

# -------- test for n-step-ahead prediction --
# sample = sys_select(sys_name,setup)
# data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm, noise=10e-4) #  data for testing, only contains data_U
# ahead_step= 100 #10#5#10#30#5#
# user_model = test
# simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='ahead', ahead_step=ahead_step)

# # # ---- varying ahead step test ------

# ahead_step_range = np.array([5,10,20,30,50,70,90, 100])
# setup.ahead_step_range = ahead_step_range
# setup.module = module
# simulation.test(user_model, user_params, data_sample_test, setup, measure=sample, condition='eval_ahead')

# # # ------ spectrum training and test ---
# spectrum_train = False
spectrum_train = True
version = 2  # name a test version
setup = init.ModelTrain_setting(sys_name)
if spectrum_train:
    sample = sys_select(sys_name, setup)
    spectrum_train_data = sample.sample_u_spectrum(change_u=setup.change_u, time_all=setup.time_spectrum_train,
                                                   norm=norm)  
    np.savetxt('ref_spectrum.txt', sample.system.ref)

    # fig, ax = plt.subplots(2, 1, sharex=True)
    # ax[0].plot(spectrum_train_data[0], 'g', label='y')
    # ax[0].legend()
    # ax[1].plot(spectrum_train_data[1], 'k', label='u')
    # ax[1].legend()
    # np.savetxt('data_spectrum_train.txt', spectrum_train_data)

    exec(f'from {module} import train')
    start_spectrum_train_time = time.time()
    user_params_spectrum = train(spectrum_train_data, setup)
    print(f"\nSpectrum_train_time_{module}: {time.time() - start_spectrum_train_time:.2f}")
    param_save(user_params_spectrum, sys_name, module,
               version=version)  #another version to note trained model different than noarmal u
if not spectrum_train:
    user_params_spectrum = param_load(sys_name, module, version)

data_sample_test = sample.sample_test(time_all=setup.time_test, norm=norm, noise=10e-6)

# np.savetxt('ref_test.txt', sample.system.ref)

exec(f'from {module} import test')
user_model = test  # using testing dataset
simulation.test(user_model, user_params_spectrum, data_sample_test, setup, measure=sample, condition='spectrum')
# np.savetxt(f'data_spectrum_yhat_{sys_name}_{module}.txt', simulation.yhat)


# # # ----------system parameters change, aging--------
# # sample = sys_select(sys_name, setup)
# # data_sample_change = sample.sample_change(change=setup.change, time_all=setup.time_test, norm=norm)  # only U
# # simulation.test(user_model, user_params, data_sample_change, setup, measure=sample, condition='dynamic')  

# # ---- no process, just read and plot ----
# # # ---- plot reference signal for EMPS -----
ref_train = np.loadtxt('ref_train.txt')
ref_test = np.loadtxt('ref_test.txt')
ref_spectrum = np.loadtxt('ref_spectrum.txt')

ticksize = 13
plt.rcParams['text.usetex']=True
plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'

fig, ax = plt.subplots(3, 1, sharex=False)
ax[0].plot(np.arange(len(ref_train)) * setup.ts, ref_train, 'r', label='train')
ax[0].legend()
ax[0].set_ylabel(r'\romannumeral 1' )  
ax[0].xaxis.set_tick_params(labelsize=ticksize)
ax[1].plot(np.arange(len(ref_test)) * setup.ts, ref_test, 'k', label='test')
ax[1].legend()
ax[1].set_ylabel(r'\romannumeral 2')
ax[1].xaxis.set_tick_params(labelsize=ticksize)
ax[2].plot(np.arange(len(ref_spectrum)) * setup.ts, ref_spectrum, 'g', label='spectrum')
ax[2].legend()
ax[2].set_ylabel(r'\romannumeral 3')
ax[2].xaxis.set_tick_params(labelsize=ticksize)
ax[2].set_xlabel('Time(s)')

test_y = np.loadtxt('data_test.txt')[0]
test_yhat = np.loadtxt(f'data_test_yhat_{sys_name}_{module}.txt')
test_yhat_spectrum = np.loadtxt(f'data_spectrum_yhat_{sys_name}_{module}.txt')
# plt.rcParams['text.usetex']=True
# plt.rcParams['text.latex.preamble']=r'\makeatletter \newcommand*{\rom}[1]{\expandafter\@slowromancap\romannumeral #1@} \makeatother'
time_exp = np.arange(len(test_y)) * setup.ts
fig, ax = plt.subplots(2, 1, sharex=False)
ax[0].plot(time_exp,test_y, 'g', label='$y$')
ax[0].plot(time_exp,test_yhat, 'r', label='$\hat{y}_0$')
ax[0].legend()
ax[0].set_ylabel(r'\romannumeral 1')
ax[0].xaxis.set_tick_params(labelsize=ticksize)
ax[1].plot(time_exp,test_y, 'g', label='$y$')
ax[1].plot(time_exp,test_yhat_spectrum, 'r', label='$\hat{y}$')
ax[1].legend()
ax[1].set_ylabel(r'\romannumeral 2')
ax[1].xaxis.set_tick_params(labelsize=ticksize)
ax[1].set_xlabel('Time(s)')

# ---- plot in paper ----
# ahead_eval_EMPS_ss = np.loadtxt('models/ahead_eval_EMPS_ss.txt')
# ahead_eval_EMPS_narmax = np.loadtxt('models/ahead_eval_EMPS_narmax.txt')
# xticksize = 14
# yticksize = 14
# legendsize = 14
# ylabelsize = 15
# xlabelsize = 17
# plot_yname = ['RMSE', 'R2', 'fit(%)']
# for i in range(3):
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(ahead_step_range, ahead_eval_EMPS_ss[:, i],'rx-', label='ss')
#     ax.plot(ahead_step_range, ahead_eval_EMPS_narmax[:, i], 'b.-', label='narmax')
#     ax.legend(fontsize=legendsize)
#     ax.set_xlabel('ahead steps')
#     ax.set_ylabel(f'{plot_yname[i]}')
#     ax.xaxis.set_tick_params(labelsize=xticksize)
#     ax.yaxis.set_tick_params(labelsize=yticksize)
#     ax.yaxis.get_label().set_fontsize(ylabelsize)
#     ax.xaxis.get_label().set_fontsize(xlabelsize)

