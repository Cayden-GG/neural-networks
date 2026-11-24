import numpy as np

# in a backpass we calculate the derivative of loss with respect to weights, bias and inputs 

dvalues = np.array([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]]) #dL_dZ the derivative of loss with respect to z or the full forward pass 

inputs = np.array([
   [1., 2., 3., 2.5], [2., 5., -1., 2], [-1.5, 2.7, 3.3, -0.8]
])

weights = np.array([[0.2, 0.8, -0.5, 1], [0.5, -0.91, 0.26, -0.5], [-0.26, -0.27, 0.17, 0.87]])

biases = np.array([[2, 3, 0.5]])

dweights = np.dot(inputs.T, dvalues)
print(f"d_w: {dweights}")

dbiases = np.sum(dvalues, axis=0, keepdims=True)
print(f"d_b: {dbiases}")

dinputs = np.dot(dvalues, weights)
print(f"d_i: {dinputs}")



