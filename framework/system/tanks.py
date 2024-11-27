import pandas as pd
import numpy as np
import math

df = pd.read_csv('F:/Project/inProcess/Framework/framework/system/tank_input.csv') #
U_train = np.array(df["uEst"]).astype(np.float32)
U_test = np.array(df["uVal"]).astype(np.float32)

N = len(U_train)


def Relu(x):  # lower tank gets additional input if upper tank overflows
    if x > 0:
        return x
    else:
        return 0


def Satu(x, satu):  # when a tank is full, reading doesn't change anymore
    if x < satu:
        return x
    else:
        return satu


class tanks2:  # open loop, 2 tanks

    def __init__(self, dt, x1=2 * 10 ** (-2), x2=2 * 10 ** (-2), v1=0, v2=0):

        self.C1 = 6 * 10 ** (-5)
        self.C2 = 7 * 10 ** (-5)
        self.A1 = 8.7 * 10 ** (-3)
        self.A2 = 5.1 * 10 ** (-3)

        self.kv = 1.1 * 10 ** (-4)

        self.h_max = 20
        self.g = 9.81

        self.dt = dt
        self.x1 = x1
        self.x2 = x2
        self.v1 = v1
        self.v2 = v2

        self.U_data = []

    def measure(self, u=0, noise=0.0, ctr=0, ref=0):
        u_max = 10
        if ctr == 1:
            u = kp * kd * (ref - self.x2) - kd * self.v2
            if u > u_max:  # Maximum voltage (saturation)
                u = u_max
            if u < 0:
                u = 0
        self.U_data.append(u)

        self.v1 = -self.C1 / self.A1 * math.sqrt(self.x1 * 2 * self.g) + self.kv / self.A1 * u + np.random.randn() *noise *0.1 #*0.1
        self.x1 = self.x1 + self.v1 * self.dt
        if self.x1 < 0:
            self.x1 = 0
        self.v2 = - self.C2 / self.A2 * math.sqrt(self.x2 * 2 * self.g) + self.C1 / self.A2 * math.sqrt(
            Satu(self.x1, self.h_max) * 2 * self.g)
        self.x2 = self.x2 + self.v2 * self.dt + Relu(self.x1 - self.h_max)+ np.random.randn() * noise
        if self.x2 < 0:
            self.x2 = 0

        self.y = Satu(self.x2, self.h_max) #

        return self.y


class measure:

    def __init__(self, dt):
        self.system = tanks2(dt)
        self.dt = dt


    def sample(self,  time_all, noise, U_train=U_train):
        """
        original system or noisy system
        """
        self.Y = []
        self.U = U_train
        for u in self.U:
            y = self.system.measure(u, noise * 10)
            self.Y.append(y)

        self.Y = np.array(self.Y, dtype=np.float32)
        return self.Y, self.U

    def sample_test(self,  time_all, noise, U_test=U_test):
        """
        # original system adding noise, non-aging
        """

        self.Y = []
        self.U = U_test

        self.U =  self.U + np.random.randn(len(self.U))*0.01

        for u in self.U:
            y = self.system.measure(u, noise * 10)
            self.Y.append(y)

        self.Y = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric

        return self.Y, self.U

    def sample_change(self, change, time_all, U_test=U_test, aging_rate=0.8):
        """
        # system parameters aging, no reference signal changing
        """

        self.Y = []
        self.U = U_test
        self.U = self.U + np.random.randn(len(self.U))*0.01
        N = int(time_all / self.dt)
        change= (change/ self.dt).astype(int)
        aging_rate = aging_rate
        noise=1e-3

        # aging starts or comment it out for original
        self.system.C1 *= aging_rate
        self.system.C2 *= aging_rate
        self.system.A1 *= aging_rate
        self.system.A2 *= aging_rate
        self.system.kv *= aging_rate

        for i in range(change[0]):  # original
            y = self.system.measure(self.U[i], noise=noise)
            self.Y.append(y)
        # aging starts
        self.system.C1 *= aging_rate
        self.system.C2 *= aging_rate
        self.system.A1 *= aging_rate
        self.system.A2 *= aging_rate
        self.system.kv *= aging_rate
        aging_rate*=0.1
        for i in range(change[0], change[1]):
            y = self.system.measure(self.U[i], noise=noise*10)
            self.Y.append(y)
        # aging
        self.system.C1 *= aging_rate
        self.system.C2 *= aging_rate
        self.system.A1 *= aging_rate
        self.system.A2 *= aging_rate
        self.system.kv *= aging_rate

        for i in range(change[1], N):
            y = self.system.measure(self.U[i], noise=10 *noise)
            self.Y.append(y)

        self.Y = np.array(self.Y, dtype=np.float32)


        return self.Y, self.U


# open loop
# ctr = 0
# U = sinwave(dt, t_max)
# U = np.array(U, dtype=np.float32)

# ------------------------------------------
#
# #  closed loop
# # kp, kd = 160.18, 243.45
# kp, kd = 0.1, 10
# ref = np.ones((N, 1))*0.15
# ctr = 1
# for i in range(int(t_max/dt)):
#     Y_sys.append(sampling.measure(ctr=ctr, ref=ref[i]))
# ----------------------------------------------------

# Y_sys= np.squeeze(Y_sys, axis=-1)


# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].plot(sampling.U_data, 'k', label='u')
# ax[0].set_xlabel('Time')
# ax[0].legend()
# ax[1].plot(Y_sys, 'g', label='y1')
# ax[1].legend()
# ax[2].plot(Y_sys, 'g', label='y2')
# ax[2].legend()
# ax[3].plot(Y_sys[:, 2], 'g', label='y3')
# ax[3].legend()
# np.savetxt(f'tanks_ctr{ctr}_t{dt}_y.txt', Y_sys)
# np.savetxt(f'tanks_ctr{ctr}_t{dt}_u.txt', sampling.U_data)
