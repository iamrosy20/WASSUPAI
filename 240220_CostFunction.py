# Look for cost functions used in regression and classification

### Regression

import numpy as np
y_true = np.array([1.0, 2.0, 3.0])
y_pred = np.array([1.1, 1.9, 3.1])

# Mean Squared Error
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
mse = mean_squared_error(y_true, y_pred)
print(mse)

# Mean Absolute Error
def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))
mae = mean_absolute_error(y_true, y_pred)
print(mae)

# Huber Loss
def huber_loss(y_true, y_pred, delta=1.0):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    squared_loss = 0.5 * (error ** 2)
    linear_loss = delta * (np.abs(error) - 0.5 * delta)
    return np.where(is_small_error, squared_loss, linear_loss)
huber = huber_loss(y_true, y_pred, delta=1.0)
print(huber)

# Log-Cosh Loss
def log_cosh_loss(y_true, y_pred):
    return np.mean(np.log(np.cosh(y_pred - y_true)))
log_cosh = log_cosh_loss(y_true, y_pred)
print(log_cosh)

# ---

### Classification

import numpy as np
from sklearn.metrics import log_loss, hinge_loss
y_true = np.array([1, 0, 1, 1, 0])
y_pred_probs = np.array([0.9, 0.1, 0.8, 0.65, 0.3])
y_pred = np.array([1, 0, 1, 1, 0])

# Cross Entropy Loss
cross_entropy = log_loss(y_true, y_pred_probs)
print(cross_entropy)

# Hinge Loss
hinge = hinge_loss(y_true, 2 * y_pred - 1)
print(hinge)
