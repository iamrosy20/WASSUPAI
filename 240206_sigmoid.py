# Plotting the sigmoid function

import numpy as np
import matplotlib.pyplot as plt

# Define simoid function
def sigmoid(z):
  return 1 / (1 + np.exp(-z))

# Define the range of input z
z = np.linspace(-10, 10, 100)

# Apply sigmoid function
sigma_z = sigmoid(z)

# Plot the sigmoid function
plt.figure(figsize=(10, 6))
plt.plot(z, sigma_z, label='Sigmoid Function')
plt.title('Sigmoid Function')
plt.xlabel('z')
plt.ylabel('sigma(z)')
plt.grid(True)
plt.legend()
plt.show()
