import numpy as np

inputs =  [[1, 2, 3, 2.5],
           [2.0, 5.0, -1.0, 2.0], 
           [-1.5, 2.7, 3.3, -0.8]] 

class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_neurons, n_inputs)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs): 
        self.output = np.dot(inputs, self.weights.T) + self.biases



layer1 = DenseLayer(4, 3)
layer1.forward(inputs)


# ReLU 
# this allows for non linear outputs 
# ReLU says if x<=0 y=0 else y=x

class ReLUActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

relu = ReLUActivation()
relu.forward(layer1.output)

layer2 = DenseLayer(3, 3)
layer2.forward(relu.output)

# softmax
# minus the max of the input from each element of the input [1 - 4, 2 - 4, 3 - 4, 4 - 4] = [-3, -2, -1, 0]
# this scales the inputs down 
# then calculate probabilities [e**1/e**1 + e**2 + e**3 + e**4]
# all outputs will sum to 1 and each element will 0<=x<=1

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

softmax = SoftmaxActivation()
softmax.forward(layer2.output)

# print(softmax.output)    

# above is a full pass without loss or optimization

# Loss Function
# the loss function used in this network will be
# Categorical Cross Entropy

# the formula is: -sum(true * log(pred))
# if true is [1, 0, 0] and output is [0.7, 0.2, 0.1] formula is -[1*log(0.7) + 0*log(0.2) + 0*log(0.1)]
# but since we only care about the correct category we the formula always simplifies to -log(pred)

class CategoricalCrossEntropyLossFunction:
    def calculate(self, pred_y, true_y):
        samples = len(pred_y)
        clipped_pred_y = np.clip(pred_y, 1e-7, 1 - 1e-7)

        correct_confidence = []
        if len(true_y.shape) == 1: # array of true indexes [1, 0, 2]
            correct_confidence = clipped_pred_y[range(samples), true_y]

        elif len(true_y.shape) == 2: # one hot encoded matrix  [[1, 0, 0], [0, 1, 0], [0, 0, 1]] 
            correct_confidence = np.sum(clipped_pred_y * true_y, axis=1)
        
        self.output = np.mean(-np.log(correct_confidence)) 

loss = CategoricalCrossEntropyLossFunction()
loss.calculate(softmax.output, np.array([1, 0, 0]))

preds = np.argmax(softmax.output, axis=1) 
accuracy = np.mean(preds == np.array([1, 0, 0]))
print(accuracy)
print(loss.output)
