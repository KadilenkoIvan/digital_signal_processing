import matplotlib.pyplot as plt
import numpy as np

def impulse(x, a=0.0, b=1.0, A=2):
    if a <= x <= b:
        return A
    return 0

def integral(a=0.0, b=1.0, A=2):
    return (np.abs(b - a)) * A

x = np.arange(-2, 2, 0.01)
y = np.array([impulse(xi) for xi in x])

integral_val = integral()
print(integral_val) 

plt.plot(x, y, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()