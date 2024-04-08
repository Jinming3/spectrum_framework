"""
framework main file, start here
"""
import time
import os
import matplotlib.pyplot as plt
# plt.style.use('F:/Project/head/tight.mplstyle')
import matplotlib
import torch
matplotlib.use("TkAgg")
import numpy as np
import sys
# sys.path.append('framework/')
from framework.framework_process import sys_select, process    #   # os.path.abspath(os.getcwd())
sys.path.append('user/')
# ----- user import >>> ------
import user_process
from user_process import ModelTrain


sys_name = 'EMPS'  # choose a system to generate data
dt = ModelTrain.dt # define the system sampling rate, setting in user_process ModelTrain
signal = 'sinwave'  # universal, 'triangel' or 'sinwave'
change1 = ModelTrain.change1
change2 = ModelTrain.change2
time_train = ModelTrain.time_train
time_test = ModelTrain.time_test  #100  # 50
sample = sys_select(sys_name, dt)

# ------------ training ---------
# data_sample_train = sample.sample(A=1, time_all=time_train, signal=signal)  # data generation for training, [Y, U]
# user_params = user_process.train(data_sample_train)
# model_folder = os.path.join("models", sys_name)
# if not os.path.exists(model_folder):
#     os.makedirs(model_folder)
# torch.save(user_params, os.path.join(model_folder, "offline"))
# # ------- or load ----
model_folder = os.path.join("models", sys_name)
user_params = torch.load(os.path.join(model_folder, "offline"))  # load saved model
simulation = process(plot=True)  # to evaluate the user model

# user_model = user_process.test
# sample = sys_select(sys_name, dt)
# data_sample_inference = sample.sample_infer(A=1, time_all=time_test, signal=signal, noise=10e-5)  # data for testing, only contains data_U
# simulation.inference(user_model, user_params, data_sample_inference, measure=sample, condition='test')

sample = sys_select(sys_name, dt)
user_model = user_process.regulator
data_sample_aging = sample.sample_aging(A=1, change1=change1, change2=change2, time_all=time_test, signal=signal, aging_rate=0.9)
simulation.inference(user_model, user_params, data_sample_aging, measure=sample, condition='dynamic', case=5)

sample = sys_select(sys_name, dt)
user_model = user_process.regulator
data_sample_ref = sample.sample_ref(A=1, change1=change1, change2=change2, time_all=time_test)  # data_sample only contains data_U. ref change
simulation.inference(user_model, user_params, data_sample_ref, measure=sample, condition='dynamic', case=5)



# #

#
# # model change to regulator
# # simulation = process(plot=True)  # to evaluate the user model
#
# data_sample_aging = sample.sample_aging(A=1, change1=change1, change2=change2, time_all=time_test, signal=signal, aging_rate=0.9)
# # np.savetxt('data_sample_aging.txt', (data_sample_aging, sample.Y_hidden),delimiter=',')
# user_model = user_process.regulator
# simulation.inference(user_model, user_params, data_sample_aging, measure=sample, condition='dynamic')
# yhat_regulator = simulation.yhat

# rate_evl = [0.99,0.97,0.95,0.93, 0.9,0.87,0.85, 0.83,0.8,0.75, 0.7, 0.65,0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# r2_evl = []
# rmse_evl=[]
# for r in rate_evl:
#
#     sample = sys_select(sys_name, dt)
#     data_sample_aging = sample.sample_aging(A=1, change1=10, change2=30, time_all=time_test, signal=signal, aging_rate=r)
#     simulation.inference(user_model, user_params, data_sample_aging, measure=sample, condition='dynamic', ahead_step=1)
#
#
#     r2_evl.append(simulation.r2)
#     rmse_evl.append(simulation.RMSE)
#
# np.savetxt(f"aging_rate_{'%.4f' %dt}.txt", [rate_evl, r2_evl, rmse_evl])
# rate_evl_plot = [1-p for p in rate_evl]
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(rate_evl_plot,r2_evl, 'k', label='$R^2$')
# ax[0].grid()
# ax[0].legend()
# ax[1].plot()
# ax[1].plot(rate_evl_plot,rmse_evl, 'k', label='RMSE')
# ax[1].set_xlabel('aging rate')
# ax[1].legend()
# ax[1].grid()

# -----------



