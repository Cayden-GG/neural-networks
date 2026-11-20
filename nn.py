import numpy as np

# each input array is a nueron
# each set (lol) if weights and biases is a layer

# number of input nuerons
inputs = [[1, 2, 3, 2.5], # 1 nueron or one set of reading at a given time
          [2.0, 5.0, -1.0, 2.0], # 2
          [-1.5, 2.7, 3.3, -0.8]] # 3

# weight set for each input
weights = [
    [0.2, 0.8, -0.5, 1.0],     # |
    [0.5, -0.91, 0.26, -0.5],  # | this is the weights for one layer
    [-0.26, -0.27, 0.17, 0.87],# |
]

# bias for each input
biases = [2, 3, 0.5] # | this is the biases for one layer

layer1_output = np.dot(inputs, np.array(weights).T) + biases

print(layer1_output)

# layer 2 of weights
weights2 = [
    [0.1, -0.14, 0.5],    # |
    [-0.5, 0.12, -0.33],  # | this is the weights for one layer
    [-0.44, 0.73, -0.13], # |
]

# layer 2 of biases
biases2 = [-1, 2, -0.5] # | this is the biases for one layer

# layer 2 inputs are layer one outputs
layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

print(layer2_output)


