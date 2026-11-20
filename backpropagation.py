import numpy as np
#
# inputs = np.array([1.0, -2.0, 3])
# weights = np.array([-3.0, -1.0, 2.0])
# bias = 1.0
#
# target_output = 0.0  # we want 0 loss
learning_rate = 0.001  # how fast we move down the curve

# backprop through 1 neuron (chain rule)
"""
# layer #
g(x) = w0 * x0 + w1 * x1 + w2 * x2 + w3 * x3 + b
z = relu(g(x))
f(x) = relu(g(x))
z = f(x)
z = max(g(x), 0) 
loss = (z - 0)**2  # square loss function (0 is the true value)
loss = z**2

Goal is to find the derivative with respect to each input w0, ... w3, b

f(g(x)) f=relu g=layer1  z=relu(g(x)) (z is output of layer) 
f'(z) * g'(x)

s=step
s0 = w0 * x0, ... s3 = w3 * x3 
=> s1 = sum(s0, ..., s3) + b 
=> s2 = relu(s1) 
=> s3 = (s2 - 0)**2

Loss = (ReLU(sum(mul(w0, x0), ..., mul(w3, x3), b)))**2

∂Loss/∂w0 = ∂loss/∂relu() * ∂relu/∂sum() * ∂sum()/∂mul(w0, x0) * ∂mul(x0, x0)/∂w0   ## this find how w0 effects the final loss

∂loss/∂relu
y = x**2
dy/dx = 2x
dy/dx = 2 * relu(sum(...))

∂relu/∂sum()
y = relu(x)
dy/dx = 1 (x > 0); 0

∂sum()/∂mul(w0, x0)
f(x,y) = x + y
df/dx = 1; df/dy = 1;
*always 1

∂mul(w0, x0)/∂w0
= x0 * ∂w0/∂w0
= x0 (x0 in this cause is 1)


∂Loss/∂w1 = ''  
∂Loss/∂w2 = ''
∂Loss/∂w3 = ''
∂Loss/∂b = ''
"""


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)
#
#
# best_loss = 100
#
# for interation in range(1000):
#     # forward pass
#     linear_output = np.dot(weights, inputs) + bias
#     output = relu(linear_output)
#     loss = (output - target_output) ** 2
#
#     # backward pass
#     dloss_doutput = 2 * (output - target_output)
#     doutput_dlinear = relu_derivative(linear_output)
#     # ∂sum()/∂mul always = 1 so left out
#     dlinear_dweights = inputs  # always equals x so inputs
#     dlinear_dbias = 1.0
#
#     dloss_dlinear = dloss_doutput * doutput_dlinear
#     dloss_dweights = dloss_dlinear * dlinear_dweights
#     dloss_dbias = dloss_dlinear * dlinear_dbias
#
#     weights -= learning_rate * dloss_dweights
#     bias -= learning_rate * dloss_dbias
#
#     if loss < best_loss:
#         best_loss = loss
#
#     print(f"Iteration: {interation + 1},  Loss: {loss}")
#
# print(f"Best Loss: {best_loss}")
# print(f"Final weights: {weights}")
# print(f"Final bias: {bias}")
#

# backprop through layer of neurons

layer_inputs = np.array([1, 2, 3, 4])
layer_weights = np.array(
    [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]]
)
layer_biases = np.array([0.1, 0.2, 0.3])

best_loss = 100000

for interation in range(200):
    # forward pass
    z = np.dot(layer_weights, layer_inputs) + layer_biases
    a = relu(z)
    y = np.sum(a)

    # loss
    loss = y ** 2

    # backward pass
    # gradient of loss with respect to output y 
    dL_dy = 2 * y

    # gradient of y with respect to a
    dL_da = np.ones_like(a)

    # loss with respect to a
    dL_da = dL_dy * dL_da

    # gradient of a with respect to z (relu derivative)
    da_dz = relu_derivative(z)

    # gradient of loss with with respect to z
    dL_dz = dL_da * da_dz

    # gradient of z with respect ro weights and biases
    dL_dW = np.outer(dL_dz, layer_inputs)
    dL_db = dL_dz

    # update weights and biases
    layer_weights -= learning_rate * dL_dW
    layer_biases -= learning_rate * dL_db

    if loss < best_loss:
        best_loss = loss

    if interation % 20 == 0:
        print(f"Iteration: {interation + 1},  Loss: {loss}")


print(f"Best Loss: {best_loss}")
print(f"Final weights: {layer_weights}")
print(f"Final bias: {layer_biases}")
