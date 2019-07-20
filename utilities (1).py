import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
# Define the minimum and maximum values for X and Y
# that will be used in the mesh grid
 min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
 min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

# Define the step size to use in plotting the mesh grid
 mesh_step_size = 0.01
# Define the mesh grid of X and Y values
 x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size),
 np.arange(min_y, max_y, mesh_step_size))
 # Run the classifier on the mesh grid
 output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])
# Reshape the output array
 print(output)
 output = output.reshape(x_vals.shape)


# Create a plot
 plt.figure()
# Choose a color scheme for the plot
 a=plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)
 plt.colorbar(a)
 plt.show()

