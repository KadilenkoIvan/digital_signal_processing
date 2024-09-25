import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

class FourierSeries:
    def __init__(self, A=2.0, T=2.0):
        self.A = A
        self.T = T
        self.w = 2 * np.pi / T

    def impulse(self, t, a=0.0, b=1.0, A=2):
        return A * np.cos(omega * t)

    def compute_a0(self):
        return (2/self.T) * quad(lambda t: self.impulse(t), 0, self.T)[0]

    def compute_an(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.cos(n * self.w * t), 0, self.T)[0]

    def compute_bn(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.sin(n * self.w * t), 0, self.T)[0]

    def fourier_series(self, t, N=10):
        return (self.compute_a0() / 2) + sum([self.compute_an(n) * np.cos(n * self.w * t) + self.compute_bn(n) * np.sin(n * self.w * t) for n in range(1, N+1)])
    
A = 2.
f = 100
T = 1/f
N = 100_000
t = np.linspace(-4, 4, N)
omega = 2 * np.pi / T

proccessor = FourierSeries(A=2., T = 1/f)

x_cos = A * np.cos(omega * t)
x_cos_my = proccessor.fourier_series(t)
x_cos_fourier = np.fft.fft(x_cos)

freqs = np.fft.fftfreq(t.size, d=t[1] - t[0])
plt.subplot(2, 1, 1)
plt.plot(t, x_cos, 'b', label='x_cos')
plt.plot(t, x_cos_my, '--r', label='x_cos_my')
plt.legend()
plt.title('Сигнал')
plt.xlabel('time')
plt.ylabel('amplitude')
plt.xlim(-0.05, 0.05)
plt.ylim(-3, 3)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(freqs[:len(freqs)//2], 2.0 / N * np.abs(x_cos_fourier)[:len(freqs)//2])
plt.title('Спектр сигнала')
plt.xlabel('freq')
plt.ylabel('amplitude')
plt.xlim(0, 200)
plt.grid(True)

plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.show()