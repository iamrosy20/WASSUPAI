# Look for some activation functions: ReLU, Sigmoid, Tanh, and SoftMax

### ReLU
import numpy as np
import matplotlib.pyplot as plt

# Define ReLU function
def relu(x):
    return np.maximum(0, x)

# Make dataset
x = np.linspace(-10, 10, 100)
y = relu(x)

# Visualize result
plt.plot(x, y, label='ReLU')
plt.title("ReLU Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# ---

### Sigmoid
import numpy as np
import matplotlib.pyplot as plt

# Define sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Make dataset
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

# Visualize result
plt.figure(figsize=(14, 6))
plt.plot(x, y, label='Sigmoid')
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# ---

### Tanh
import numpy as np
import matplotlib.pyplot as plt

# Define Tanh function
def tanh(x):
    return np.tanh(x)

# Make dataset
x = np.linspace(-10, 10, 100)
y = tanh(x)

# Visualize result
plt.figure(figsize=(14, 6))
plt.plot(x, y, label='Tanh')
plt.title("Tanh Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.legend()
plt.show()

# ---

### SoftMax
import numpy as np

# Define SoftMax function
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()

# Make dataset
x = np.array([2.0, 1.0, 0.1])
y = softmax(x)

# Print result
print(y)
