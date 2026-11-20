import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

learning_rate = 0.001

X, y = spiral_data(samples=100, classes=3) 

class DenseLayer:
    def __init__(self, n_neurons, n_inputs):
        self.weights = np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros(n_neurons)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights.T) + self.biases

    def backward(self):
        pass

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(inputs, 0)

    def backward(self):
        pass

layer = DenseLayer(3, 2)
layer.forward(X)

print(layer.output[:5])
