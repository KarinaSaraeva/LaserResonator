import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd

M = 100
a = np.empty([M, 3])
freq_0 = 10000
d_v = 26  # MHz
d_v_gen = 1092  # MHz

counts = 20000
time = 120  # ns
resonator_time = 38.04  # time of resonator is approximately 38 ns
period = math.floor(counts / (2 * time / resonator_time))


def ampl_of_mode(numb):
    freq = freq_0 + 1 / 2 * (2 * numb + 1 - M) * 2 * math.pi * d_v
    ampl = 1 / math.cosh(2 * math.acosh(2) * (freq - freq_0) / (2 * math.pi * d_v_gen))
    phase = random.uniform(-math.pi, math.pi)
    return freq, ampl, phase


for i in range(M):
    a[i, :] = ampl_of_mode(i)

E = np.empty([counts, 2])

for k in range(counts):
    E[k, 0] = k * (time / counts)
    E[k, 1] = 0
    for i in range(M):
        E[k, 1] = E[k, 1] + a[i, 1] * math.cos(a[i, 0] * E[k, 0] * (10 ** (-3)) + a[i, 2])

df = pd.DataFrame(E[:, 1])
df.to_csv(r'D:\SHITTT.txt', header=None, index=None, sep=' ', mode='a')

Intensity = (E[:, 1]) ** 2
# averaging
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid')

Averaged = moving_average(Intensity, period)

fig, axs = plt.subplots(2, 2)
axs[0, 0].scatter(a[:, 0], a[:, 1])
axs[0, 0].set_title('amplitudes')
axs[0, 1].scatter(a[:, 0], a[:, 2])
axs[0, 1].set_title('amplitudes')
axs[1, 0].plot(E[:, 0], Intensity)
axs[1, 0].set_title('intensity')
axs[1, 1].plot(np.linspace(0, 120, len(Averaged)), Averaged)
axs[1, 1].set_title('resonator half-time averaged inensity')

for ax in axs.flat:
    ax.label_outer()

plt.show()