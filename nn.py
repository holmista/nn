import numpy as np
import pandas as pd


def forward(x: np.ndarray, w: np.ndarray, activation=None):
    """
    Computes forward pass for a fully connected layer
    if activation function is provided it will be applied

    Parameters:
    - x: A numpy ndarray(1 x n), activations of previous layer
    - w: A numpy ndarray, weights connecting this layer to the next layer

    Returns:
    - nd numpy array(1 x m)
    """

    forward_x = x.dot(w).reshape(1, -1)
    activated_x = relu_activation(forward_x)

    return forward_x, activated_x


def backward(
    dprev: np.ndarray,
    activated: np.ndarray,
    unactivated: np.ndarray,
    w: np.ndarray,
):
    """
    Computes backward pass for a fully connected layer

    Parameters:
    - dprev: A numpy nd array of previous gradient(1 x n)
    - activated: A numpy nd array of layer with activated values(1 x m)
    - unactivated: the same as x except activation is not applied
    - w: A numpy nd array of weights connecting this layer to the previous layer

    Returns:
    - dcdx: gradient with respect to x(1 x k)
    - dcdw: gradient with respect to w
    """

    rd = relu_derivative(unactivated)
    dcdx = dprev.dot(w.T) * rd

    dcdw = activated.T.dot(dprev)

    return dcdx, dcdw


def relu_activation(x: np.ndarray):
    """
    Applies relu activation on an input

    Parameters:
    - x: A numpy nd array(1 x n)

    Returns:
    - activated_x: A numpy nd array(1xn)
    """

    activated_x = np.maximum(0, x)
    return activated_x


def relu_derivative(x: np.ndarray):
    """
    Computes the derivative of the relu activation function.

    Parameters:
    - x: A numpy nd array

    Returns:
    - derivative: The derivative of relu applied to x.
    """

    mask = x > 0

    derivative = x * mask
    return derivative


def create_fully_connected_weights(layer1_length: int, layer2_length: int):
    """
    Creates weights between two layers and initializes them to random values between 0 and 1
    The layers are assumed to be fully connected
    The column of the returned weights need to be multiplied by previous layer to get outputs of layer 2

    Parameters:
    - layer1_length
    - layer2_length

    Returns:
    - np array of shape m x n. m - length of layer 1, n - length of layer 2
    """

    m = layer1_length
    n = layer2_length

    return np.random.rand(m, n)


def create_layer(n: int):
    """
    Creates and returns a zero numpy array of shape 1 x n

    Parameters:
    - n amount of nodes in a layer
    """
    return np.zeros(n).reshape(1, -1)


input_layer = create_layer(4)
hidden_layer = create_layer(2)
output_layer = create_layer(1)

weights1 = create_fully_connected_weights(input_layer.size, hidden_layer.size)
weights2 = create_fully_connected_weights(hidden_layer.size, output_layer.size)

X_train = np.array(
    [[1, 2, 3, 4], [1.5, 2, 2.4, 3.5], [3.2, 0.7, 1.5, 2.1], [-0.5, 1.6, -1, 4]]
)
y_train = np.array([2.25, 2.725, 2.675, 0.9])

X_test = np.array([[2, 3, 4, 5], [-1, 2, 3, 4]])
y_test = np.array([4, 1.75])


learning_rate = 0.01

for epoch in range(70):
    total_cost = 0
    # do forward and backward propagation for each input row
    for i in range(X_train.shape[0]):
        # forward
        input_layer = X_train[i].reshape(1, -1)
        hidden_layer, hidden_layer_activated = forward(input_layer, weights1)
        output_layer, output_layer_activated = forward(hidden_layer_activated, weights2)

        cost = (output_layer[0][0] - y_train[i]) ** 2
        total_cost += cost

        # backward
        dcdl_output = 2 * (output_layer - y_train[i].reshape(1, -1)) * 1
        dcdl_hidden, dcdw2 = backward(
            dcdl_output, hidden_layer_activated, hidden_layer, weights2
        )
        dcdl_input, dcdw1 = backward(dcdl_hidden, input_layer, input_layer, weights1)

        weights1 -= learning_rate * dcdw1
        weights2 -= learning_rate * dcdw2

    print(total_cost)


def predict(x: np.ndarray, weights1: np.ndarray, weights2: np.ndarray):
    # do forward propagation
    input_layer = x
    hidden_layer, hidden_layer_activated = forward(input_layer, weights1)
    output_layer, output_layer_activated = forward(hidden_layer_activated, weights2)

    return output_layer[0][0]


def calculate_error(X: np.ndarray, y: np.array, weights1, weights2):
    """
    Parameters:

    - X: nd numpy array of features (m x n)
    - y: numpy array
    """

    err = 0
    for i in range(X.shape[0]):
        prediction = abs(predict(X[i].reshape(1, -1), weights1, weights2) - y[i])
        actual = y[i]

        diff = abs(actual - prediction)
        err += diff**2

    return err / X.shape[0]


print(calculate_error(X_test, y_test, weights1, weights2))
