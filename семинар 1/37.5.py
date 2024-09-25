import matplotlib.pyplot as plt
import numpy as np

def teylor(x, n, out=False):
    y = 0
    for k in range(n):
        y += (x**k)/np.math.factorial(k)
        if out:
            print(y)
    return y

#a
print(f'exp: {np.exp(1.5)}')
print(f'5  : {teylor(1.5, 5)}')
print(f'6  : {teylor(1.5, 6)}')
print(f'7  : {teylor(1.5, 7)}')
print()

#b
print(f'exp: {np.exp(1.5)}')
teylor(1.5, 7, True)

x = np.arange(-1, 3, 0.01)
y = np.exp(x)

y5 = teylor(x, 5)
y6 = teylor(x, 6)
y7 = teylor(x, 7)

#plt.subplot(1, 2, 2)
plt.plot(x, y, 'r', x, y5, '--g', x, y6, '--b', x, y7, '--y')

plt.grid(True)
plt.legend()
plt.show()