# Plotting the sigmoid function

import numpy as np
import matplotlib.pyplot as plt

x = np.arrange(-6, 6, 0.1)
phi = 1 / (1 + np.exp(-x))

plt.plot(x, phi)
plt.xlabel('x')
plt.ylabel('phi')

plt.show()
