import numpy as np
import math
from scipy import signal

np.random.seed(7)


def inductance(il, L0):
    out = L0 * (0.9 * (1 / math.pi * np.arctan(-5 * (np.abs(il) - 5)) + 0.5) + 0.1)
    return out


def white(bandwidth, time_all, std_dev, ts):  # Sample rate in Hz # Duration of the white noise in seconds
    fs_noise = 2 * bandwidth  # Noise generation sampling frequency, should be at least twice the bandwidth
    t_noise = np.arange(0, time_all, 1 / fs_noise)
    noise = std_dev * np.random.randn(len(t_noise))
    num_samples = int(time_all / ts)
    sampled_noise = signal.resample(noise, num_samples)
    return sampled_noise


class rlc:
    def __init__(self, ts, vc=0, il=0, dvc=0, dil=0):
        self.vc = vc  # capacitor voltage (V)
        self.il = il  # inductor current (A)
        self.dvc = dvc  # derivative
        self.dil = dil  # derivative
        self.ts = ts

    def measure(self, u, noise_process=0.0, noise_measure=0.0, L0=50*10**(-6), C=270*10**(-9), R=3):
        self.u = u
        self.L = inductance(self.il, L0)
        self.dvc = self.il / C
        self.dil = -1 / self.L * self.vc - R / self.L * self.il + 1 / self.L * u + np.random.randn() * noise_process
        self.vc = self.vc + self.dvc * self.ts + np.random.normal(0, 10) * noise_measure
        self.il = self.il + self.dil * self.ts + np.random.normal(0, 1) * noise_measure
        output = self.vc
        return output


class measure:

    def __init__(self, ts):
        self.system = rlc(ts)
        self.ts = ts

    def sample(self, time_all, noise):
        """
        original system or noisy system
        """
        self.Y = []
        self.U = []
        self.X = []

        bandwidth = 300e2
        std_dev = 80
        v_in = white(bandwidth, time_all, std_dev, self.ts)

        for u in v_in:
            y = self.system.measure(u, noise, noise)
            self.Y.append(y)
            self.U.append(self.system.u)
            self.X.append([self.system.vc, self.system.il])

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)
        self.X = np.array(self.X, dtype=np.float32)
        return self.Y, self.U

    def sample_test(self, time_all, noise):
        """
        original system adding noies, non-changing
        """

        self.Y = []
        self.U = []
        bandwidth = 300e2
        std_dev = 80
        v_in = white(bandwidth, time_all, std_dev, self.ts)

        for u in v_in:
            y = self.system.measure(u, noise * 10, noise)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)  # not return to user, only for framework metric
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U

    def sample_change(self, change, time_all):

        self.Y = []
        self.U = []
        change = (change/self.ts).astype(int)
        N = int(time_all / self.ts)
        bandwidth = 300e2
        std_dev = 80
        v_in = white(bandwidth, time_all, std_dev, self.ts)

        for i in range(change[0]):  # first section original
            y = self.system.measure(v_in[i], noise_process=1e-7, noise_measure=1e-8)
            self.Y.append(y)
            self.U.append(self.system.u)

        # changes start
        bandwidth = 350e2
        std_dev = 60
        v_in = white(bandwidth, time_all, std_dev, self.ts)
        for i in range(change[0], change[1]):
            y = self.system.measure(v_in[i], noise_process=1e-1, noise_measure=1e-3, L0=40 * 10 ** (-6),
                                    C=170 * 10 ** (-9), R=7)
            self.Y.append(y)
            self.U.append(self.system.u)
        # change again
        bandwidth = 100e2
        std_dev = 70
        v_in = white(bandwidth, time_all, std_dev, self.ts)

        for i in range(change[1], change[2]):
            y = self.system.measure(v_in[i], noise_process=1e-2, noise_measure=1e-4, L0=30 * 10 ** (-6),
                                    C=100 * 10 ** (-9), R=14)
            self.Y.append(y)
            self.U.append(self.system.u)

        bandwidth = 200e2  # 150e3
        std_dev = 30
        v_in = white(bandwidth, time_all, std_dev, self.ts)
        for i in range(change[2], N):
            y = self.system.measure(v_in[i], noise_process=1, noise_measure=1e-3, L0=20 * 10 ** (-6),
                                    C=70 * 10 ** (-9), R=17)
            self.Y.append(y)
            self.U.append(self.system.u)

        self.Y = np.array(self.Y, dtype=np.float32)
        self.U = np.array(self.U, dtype=np.float32)

        return self.Y, self.U


