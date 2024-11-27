import numpy as np
import math



def sinwave(dt, i, w, A, sig=1, phi=0):
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


def triangle(dt, i, A=1):
    """
    triangle wave signal
    A: amplitude
    p: period
    """
    p = 8
    x = A * 2 * np.abs(i * dt / p - math.floor(k * dt / p + 0.5))

    return x


class Spring2:
    def __init__(self, dt, pos1=0, pos2=0, vel1=0, vel2=0, acc1=0, acc2=0, m1=20, m2=20, k1=1000, k2=2000, d1=1, d2=5,
                 Fc=0.05):
        self.pos1 = pos1
        self.pos2 = pos2
        self.vel1 = vel1
        self.vel2 = vel2
        self.acc1 = acc1
        self.acc2 = acc2
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2
        self.d1 = d1
        self.d2 = d2
        self.Fc = Fc
        self.dt = dt

    def measure(self, ref, noise_process=0.0, noise_measure=0.0):
        self.u = ref
        self.acc1 = -(self.k1 + self.k2 / self.m1) * self.pos1 - (
                    self.d1 + self.d2) / self.m1 * self.vel1 + self.k2 / self.m1 * self.pos2 + self.d2 / self.m1 * self.vel2 - self.Fc / self.m1 * np.sign(
            self.vel1)
        self.acc2 = self.k2 / self.m2 * self.pos1 + self.d2 / self.m2 * self.vel1 - self.k2 / self.m2 * self.pos2 - self.d2 / self.m2 * self.vel2 + 1 / self.m2 * self.u - self.Fc / self.m2 * np.sign(
            self.vel2) + np.random.randn() * noise_process
        self.vel1 = self.vel1 + self.acc1 * self.dt
        self.vel2 = self.vel2 + self.acc2 * self.dt
        self.pos1 = self.pos1 + self.vel1 * self.dt
        self.pos2 = self.pos2 + self.vel2 * self.dt
        output = self.pos2 + np.random.randn() * noise_measure
        return output


class measure:

    def __init__(self, ts):
        self.system = Spring2(ts)
        self.ts = ts

    def sample(self, time_all,  noise=0):
        """
        original system or noisy system
        """
        self.Y = []
        self.U = []

        for i in range(int(time_all/self.ts)):
            y = self.system.measure(sinwave(dt=self.ts, i=i, w=0.5, A=4), noise * 10, noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)
        return self.Y, self.U

    def sample_test(self,time_all, noise=10e-3):
        """
        # original system adding noise, non-aging
        """

        self.Y = []
        self.U = []

        for i in range(int(time_all/self.ts)):
            y = self.system.measure(sinwave(dt=self.ts, i=i, w=0.5, A=4), noise * 10, noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U


    def sample_change(self, change, time_all):
        self.Y = []
        self.U = []

        N = int(time_all / self.ts)
        change = (change/self.ts).astype(int)
        for i in range(change[0]):  # first section in original condition
            y = self.system.measure(sinwave(dt=self.ts, i=i, w=0.5, A=3), noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.system.m1 = self.system.m1 * 0.98
        self.system.m2 = self.system.m2 * 0.98
        self.system.k1 = self.system.k1 * 0.96
        self.system.k2 = self.system.k2 * 0.98
        self.system.d1 = self.system.d1 * 0.96
        self.system.d2 = self.system.d2 * 0.98

        for i in range(change[0], change[1]):
            y = self.system.measure(sinwave(dt=self.ts, i=i, w=0.5, A=1.5), noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[1], change[2]):
            y = self.system.measure(ref=1, noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.system.m1 = self.system.m1 * 0.95
        self.system.m2 = self.system.m2 * 0.98
        self.system.k1 = self.system.k1 * 0.96
        self.system.k2 = self.system.k2 * 0.96
        self.system.d1 = self.system.d1 * 0.99
        self.system.d2 = self.system.d2 * 0.98
        for i in range(change[2], change[3]):
            y = self.system.measure(ref=-1, noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        for i in range(change[3], change[4]):
            y = self.system.measure(ref=sinwave(self.ts, i, 1/2, 0.6)+sinwave(self.ts, i, 1/3, 0.4), noise_process=1e-4, noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.system.m1 = self.system.m1 * 0.98
        self.system.m2 = self.system.m2 * 0.97
        self.system.k1 = self.system.k1 * 0.98
        self.system.k2 = self.system.k2 * 0.96
        self.system.d1 = self.system.d1 * 1.99
        self.system.d2 = self.system.d2 * 1.98
        for i in range(change[4], change[5]):
            y = self.system.measure(ref=sinwave(self.ts, i, 0.1, 0.6), noise_process=1e-4,
                                 noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.system.m1 = self.system.m1 / math.sqrt(self.system.m1)
        self.system.m2 = self.system.m2 * math.sqrt(self.system.m2)
        self.system.k1 = self.system.k1 - math.sqrt(self.system.k1)
        self.system.k2 = self.system.k2 - self.system.k2 ** (-2)
        self.system.d1 = self.system.d1 * 1.99
        self.system.d2 = self.system.d2 * 1.98
        for i in range(change[5], N):
            y = self.system.measure(ref=sinwave(dt=self.ts, i=i, w=0.5, A=2), noise_process=1e-4,
                                 noise_measure=1e-5)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U











