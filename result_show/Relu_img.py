import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.maximum(0.1*x, x)

x = np.linspace(-10, 10, 100)
y_relu = relu(x)
y_leaky_relu = leaky_relu(x)

plt.subplots(1, 2, figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.plot(x, y_relu)
plt.title('ReLU')
plt.subplot(1, 2, 2)
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
plt.plot(x, y_leaky_relu)
plt.title('Leaky ReLU')
plt.show()
