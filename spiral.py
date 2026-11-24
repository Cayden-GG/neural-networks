import numpy as np
import nnfs 
from nnfs.datasets import spiral_data

nnfs.init()

X, y = spiral_data(samples=100, classes=3) 

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues) 
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class CategoricalCrossEntropyLossFunction(Loss):
    def forward(self, pred_y, true_y):
        samples = len(pred_y)
        clipped_pred_y = np.clip(pred_y, 1e-7, 1 - 1e-7)

        correct_confidence = []
        if len(true_y.shape) == 1: # array of true indexes [1, 0, 2]
            correct_confidence = clipped_pred_y[range(samples), true_y]

        elif len(true_y.shape) == 2: # one hot encoded matrix  [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
            correct_confidence = np.sum(clipped_pred_y * true_y, axis=1)
        
        negative_log_likelihood = -np.log(correct_confidence)
        return negative_log_likelihood

    def backward(self, dvalues, true_y):
        samples = len(dvalues)
        lables = dvalues[0]

        if len(true_y.shape) == 1:
            true_y = np.eye(lables)[true_y]
        
        self.dinputs = -true_y / dvalues
        self.dinputs = self.dinputs / samples
            

class ActivationSoftmaxLossCategoricalCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = CategoricalCrossEntropyLossFunction()

    def forward(self, inputs, true_y):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.calculate(self.output, true_y)

    def backward(self, dvalues, true_y):
        samples = len(dvalues)

        if len(true_y.shape) == 2:
            true_y = np.argmax(true_y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), true_y] -= 1

        self.dinputs = self.dinputs / samples


class OptimizerSGD:
    def __init__(self, learning_rate=1, decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weights_update = self.momentum * layer.weight_momentums - \
                             self.current_learning_rate * layer.dweights
            layer.weight_momentums = weights_update

            bias_update = self.momentum * layer.bias_momentums - \
                             self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_update

        else:
            weights_update = -self.learning_rate * layer.dweights
            bias_update = -self.learning_rate * layer.dbiases

        layer.weights += weights_update
        layer.biases += bias_update

    def post_update_params(self):
        self.iterations += 1

class RMSPropOptimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        layer.weight_cache = self.rho * layer.weight_cache + \
            (1 - self.rho) * layer.dweights**2

        layer.bias_cache = self.rho * layer.bias_cache + \
            (1 - self.rho) * layer.dbiases**2

        layer.weights += -self.current_learning_rate * layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)

        layer.biases += -self.current_learning_rate * layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


class AdamOptimizer:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentum
        # self.iteration is 0 at first pass and we need to start with 1 here    
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2

        # Get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Vanilla SGD parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

        layer.weights += -self.current_learning_rate * \
            layer.dweights / \
            (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
            layer.dbiases / \
            (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


layer1 = DenseLayer(2, 64)
activation1 = ReLU()

layer2 = DenseLayer(64, 3)
loss_activation = ActivationSoftmaxLossCategoricalCrossEntropy()

# optimizer = RMSPropOptimizer(learning_rate=0.02, decay=1e-5, rho=0.999)
# optimizer = OptimizerSGD(decay=1e-7, learning_rate=1, momentum=0.9)
optimizer = AdamOptimizer(decay=1e-5,  learning_rate=0.02)

for epoch in range(100_001):
    # forward
    layer1.forward(X)
    activation1.forward(layer1.output)

    layer2.forward(activation1.output)
    loss = loss_activation.forward(layer2.output, y)
    
    # outputs
    pred = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)

    accuracy = np.mean(pred == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}')

    # backward
    loss_activation.backward(loss_activation.output, y)
    layer2.backward(loss_activation.dinputs)
    activation1.backward(layer2.dinputs)
    layer1.backward(activation1.dinputs)

    # optimize
    optimizer.pre_update_params()
    optimizer.update_params(layer1)
    optimizer.update_params(layer2)
    optimizer.post_update_params()
