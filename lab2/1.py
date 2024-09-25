import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

def DFT_slow(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

A = 2.
f1 = 150
f2 = 50
T1 = 1/f1
T2 = 1/f2
N = 1_000
omega1 = 2 * np.pi / T1
omega2 = 2 * np.pi / T2

t = np.linspace(-0.05, 0.05, N)
x_cos = A * np.cos(omega1 * t) + A * np.cos(omega2 * t)

x_cos_DFT_slow = DFT_slow(x_cos)
x_cos_fft = np.fft.fft(x_cos)
freqs = np.fft.fftfreq(t.size, d=t[1] - t[0])

plt.figure(figsize=(8,8))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

plt.subplot(3, 1, 1)
plt.plot(t, x_cos)
plt.title('Сигнал')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.xlim(-0.05, 0.05)
plt.grid(True)

plt.subplot(3,1,2)
plt.plot(freqs[:len(freqs)//2], 2.0 / N * np.abs(x_cos_DFT_slow)[:len(freqs)//2])
plt.title('Спектр сигнала')
plt.xlabel('freq')
plt.ylabel('amplitude')
plt.xlim(0, 200)
plt.grid(True)

plt.subplot(3,1,3)
plt.plot(freqs[:len(freqs)//2], 2.0 / N * np.abs(x_cos_fft)[:len(freqs)//2])
plt.title('Спектр сигнала')
plt.xlabel('freq')
plt.ylabel('amplitude')
plt.xlim(0, 200)
plt.grid(True)

plt.show()