# Visualization of vectors

import matplotlib.pyplot as plt

# Make vectors
a = (1, 2)
b = (2, 2)
c = (-3, -3)

# Draw coordinate plane
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel('X axis')
plt.ylabel('Y axis')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Draw the vectors
plt.quiver(0, 0, a[0], a[1], angles='xy', scale_units='xy', scale=1, color='r', label='a[1, 2]')
plt.quiver(0, 0, b[0], b[1], angles='xy', scale_units='xy', scale=1, color='g', label='b[2, 2]')
plt.quiver(0, 0, c[0], c[1], angles='xy', scale_units='xy', scale=1, color='b', label='c[-3, -3]')

# Plot the result
plt.legend()
plt.title('2D Vector Visualization')
plt.show()
