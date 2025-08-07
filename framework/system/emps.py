import numpy as np
import math

np.random.seed(3)


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
        self.satu = satu # control voltage u limit
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


def sinwave(A, dt, i, w, sig=1, phi=0):
    """
    sin/cos wave signal
    w: frequency
    A: amplitude
    """

    if sig == 1:
        x = A * np.cos(w * i * math.pi * dt + phi)
    if sig == 0:
        x = A * np.sin(w * i * math.pi * dt + phi)

    return x


def triangle(A, i, dt, p=2.5):  
    """
    triangle wave signal
    A: amplitude
    p: period
    """

    x = A * 2 * np.abs(i * dt / p - math.floor(i * dt / p + 0.5))

    return x


class measure:

    def __init__(self, dt):
        self.system = EMPS(dt)
        self.dt = dt

    def sample(self, time_all, noise=0):
        """
        original system and/not with noise
        """
        self.Y = []
        self.U = []
        self.ref = []  # only for demonstration

        for i in range(int(time_all / self.dt)):
            pos_ref = sinwave(1, self.dt, i, w=0.7)
            y = self.system.measure(pos_ref, noise * 10, noise)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U

    def sample_test(self, time_all, noise=1e-6):  
        """
         original system adding noise, non-aging
        """

        self.Y = []
        self.U = []
        self.ref = []  # only for demonstration

        for i in range(int(time_all / self.dt)):
            pos_ref = sinwave(1, self.dt, i, 0.5)  
            y = self.system.measure(pos_ref, noise * 10, noise)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)  # not return to user, only for testing framework metric
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U


    def sample_u_spectrum(self, change_u, time_all):
        """
        system input change spectrum, test model robustness
        :param change_u: the time u spectrum change
        :param time_all: overall time
        :return:
        """

        N = int(time_all / self.dt) # overall data size
        change = (change_u / self.dt).astype(int) # changing data point

        self.Y = []
        self.U = []
        self.ref = []

        for i in range(change[0]):  # original condition adding noise
            pos_ref = sinwave(1.2, self.dt, i, 0.3) # 0.5
            y = self.system.measure(pos_ref, noise_process=1e-6, noise_measure=1e-5)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[0], change[1]):
            pos_ref = sinwave(0.8, self.dt, i, 0.5)
            y = self.system.measure(pos_ref, noise_process=1e-6, noise_measure=1e-5)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[1], change[2]):
            pos_ref = triangle(1, i, self.dt)  # sinwave(1.0, self.dt, i, 0.8)#
            y = self.system.measure(pos_ref, noise_process=1e-4, noise_measure=1e-5)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[2], change[3]):
            pos_ref = sinwave(0.6, self.dt, i, 1/2)+sinwave(0.4, self.dt, i, 1/3)
            y = self.system.measure(pos_ref, noise_process=1e-4, noise_measure=1e-5)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[3], N):  # parameter is same, reference signal changes
            pos_ref = sinwave(1.2, self.dt, i, 0.7) # 0.7
            y = self.system.measure(pos_ref, noise_process=1e-3, noise_measure=1e-4)  # triangle(2, i, self.dt)
            self.ref.append(pos_ref)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U

    def sample_change(self, change, time_all, aging_rate=0.80):
        """
        system under ambient impacts, test model robustness
        :param change: the time system parameters change
        :param time_all: overall time
        :param aging_rate:
        :return:
        """
        aging_rate=aging_rate
        N = int(time_all / self.dt) # overall data size
        change = (change / self.dt).astype(int) # changing data point

        self.Y = []
        self.U = []

        for i in range(change[0]):  # original condition adding noise
            y = self.system.measure(sinwave(1, self.dt, i, 0.5), noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.system.offset = self.system.offset * 0.99
        self.system.M = self.system.M * aging_rate
        self.system.Fv = self.system.Fv * aging_rate
        self.system.Fc = self.system.Fc * aging_rate

        for i in range(change[0], change[1]):
            y = self.system.measure(sinwave(1, self.dt, i, 0.5), noise_process=1e-3, noise_measure=1e-4)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[1], N):  # parameter is same, reference signal changes
            y = self.system.measure(triangle(2, i, self.dt), noise_process=1e-3, noise_measure=1e-4)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U



