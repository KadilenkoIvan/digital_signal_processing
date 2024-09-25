import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

class FourierSeries:
    def __init__(self, A=2.0, T=2.0):
        self.A = A
        self.T = T
        self.w = 2 * np.pi / T

    def impulse(self, t, a=0.0, b=1.0, A=2):
        if a <= t % 2 <= b:
            return A
        return -1

    def compute_a0(self):
        return (2/self.T) * quad(lambda t: self.impulse(t), 0, self.T)[0]

    def compute_an(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.cos(n * self.w * t), 0, self.T)[0]

    def compute_bn(self, n):
        return (2/self.T) * quad(lambda t: self.impulse(t) * np.sin(n * self.w * t), 0, self.T)[0]

    def fourier_series(self, t, N=10):
        return (self.compute_a0() / 2) + sum([self.compute_an(n) * np.cos(n * self.w * t) + self.compute_bn(n) * np.sin(n * self.w * t) for n in range(1, N+1)])
    
proccessor = FourierSeries()
t = np.arange(-4., 4., 0.01)
y = np.array([proccessor.impulse(ti) for ti in t])
y_fourier = proccessor.fourier_series(t)

plt.subplot(2,1,1)
plt.title('Импульсы и приближение Фурье')
plt.plot(t, y, 'r', t, y_fourier, 'g')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.subplot(2,1,2)
plt.title('Ошибка приближения Фурье')
plt.plot(t, y - y_fourier, 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

plt.show()