import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

class FourierSeries:
    def __init__(self, A=2.0, T=2.0):
        self.A = A
        self.T = T
        self.w = 2 * np.pi / T

    def impulse(self, t, a=0.0, b=1.0, A=2, noize_scale=0.0):
        if a <= t % 2 <= b:
            return A + noize_scale * np.random.rand()
        return -1 + noize_scale * np.random.rand()
    

    def compute_a0(self):
        return (2/self.T) * quad(lambda t: self.impulse(t), 0, self.T)[0]

    def compute_an(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.cos(n * self.w * t), 0, self.T)[0]

    def compute_bn(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.sin(n * self.w * t), 0, self.T)[0]

    def fourier_series(self, t, N=10):
        return (self.compute_a0() / 2) + sum([self.compute_an(n) * np.cos(n * self.w * t) + self.compute_bn(n) * np.sin(n * self.w * t) for n in range(1, N+1)])
    
proccessor = FourierSeries()
N = 600
t = np.linspace(-4, 4, N)
y_my = np.array([proccessor.impulse(ti) for ti in t])
y_fourier_my = proccessor.fourier_series(t)

y_fft_fourier = np.fft.fft(y_fourier_my)
y_fft_impulse = np.fft.fft(y_my)

freqs_fourier = np.fft.fftfreq(t.size, d=t[1] - t[0])
freqs_impulse = np.fft.fftfreq(t.size, d=t[1] - t[0])

plt.figure(figsize=(12, 6))

plt.subplot(3,2,1)
plt.title('импульс и приближение Фурье')
plt.plot(t, y_my, 'r', t, y_fourier_my, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3,2,3)
plt.title('спектр прямоугольного')
plt.plot(freqs_impulse[:len(freqs_impulse)//2], 2.0 / N * np.abs(y_fft_impulse)[:len(freqs_impulse)//2])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 60)
plt.grid(True)

plt.subplot(3,2,5)
plt.title('спектр приближения')
plt.plot(freqs_fourier[:len(freqs_fourier)//2], 2.0 / N * np.abs(y_fft_fourier)[:len(freqs_fourier)//2])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 60)
plt.grid(True)

proccessor = FourierSeries()
t = np.linspace(-4, 4, N)
y_my = np.array([proccessor.impulse(ti, noize_scale=0.5) for ti in t])
y_fourier_my = proccessor.fourier_series(t)

y_fft_fourier = np.fft.fft(y_fourier_my)
y_fft_impulse = np.fft.fft(y_my)

freqs_fourier = np.fft.fftfreq(t.size, d=t[1] - t[0])
freqs_impulse = np.fft.fftfreq(t.size, d=t[1] - t[0])

plt.subplot(3,2,2)
plt.title('импульс и приближение Фурье с шумом')
plt.plot(t, y_my, 'r', t, y_fourier_my, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(3,2,4)
plt.title('спектр прямоугольного с шумом')
plt.plot(freqs_impulse[:len(freqs_impulse)//2], 2.0 / N * np.abs(y_fft_impulse)[:len(freqs_impulse)//2])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 60)
plt.grid(True)

plt.subplot(3,2,6)
plt.title('спектр приближения с шумом')
plt.plot(freqs_fourier[:len(freqs_fourier)//2], 2.0 / N * np.abs(y_fft_fourier)[:len(freqs_fourier)//2])
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, 60)
plt.grid(True)

plt.subplots_adjust(wspace=0.6, hspace=0.6)
plt.show()