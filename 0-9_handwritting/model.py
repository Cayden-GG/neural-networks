import numpy as np
import idx2numpy
import random

data = np.load("models/0-9handwriting_model.npz")
params = {key: data[key] for key in data.files}

test_images_file_path = "data/t10k-images.idx3-ubyte"
test_lables_file_path = "data/t10k-labels.idx1-ubyte"

images = idx2numpy.convert_from_file(test_images_file_path)
images = images.reshape(images.shape[0], -1)
labels = idx2numpy.convert_from_file(test_lables_file_path)

class DenseLayer:
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

layer1 = DenseLayer(params["w1"], params["b1"])
relu = ReLU()
layer2 = DenseLayer(params["w2"], params["b2"])
softmax = Softmax()

selection = random.randint(0, 9_999)
layer1.forward(images[selection])
relu.forward(layer1.output)
layer2.forward(relu.output)
softmax.forward(layer2.output)

pred = np.argmax(softmax.output)

print(f"Model Prediction: {pred}\nTrue Value: {labels[selection]}")


