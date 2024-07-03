import numpy as np

class NN:

    def __init__(self):
        self.weights = []

    def forward(self, x: np.ndarray, w: np.ndarray, activation=None):
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
        activated_x = self.relu_activation(forward_x)

        return forward_x, activated_x


    def backward(
        self,
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

        rd = self.relu_derivative(unactivated)
        dcdx = dprev.dot(w.T) * rd

        dcdw = activated.T.dot(dprev)

        return dcdx, dcdw


    def relu_activation(self, x: np.ndarray):
        """
        Applies relu activation on an input

        Parameters:
        - x: A numpy nd array(1 x n)

        Returns:
        - activated_x: A numpy nd array(1xn)
        """

        activated_x = np.maximum(0, x)
        return activated_x


    def relu_derivative(self, x: np.ndarray):
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


    def create_fully_connected_weights(self, layer1_length: int, layer2_length: int):
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

        weights = np.random.rand(m, n)
        return weights


    def create_layer(self, n: int):
        """
        Creates and returns a zero numpy array of shape 1 x n

        Parameters:
        - n amount of nodes in a layer
        """
        layer = np.zeros(n).reshape(1, -1)
        return layer
    
    def train(self, epochs: int, X_train: np.ndarray, y_train: np.ndarray, learning_rate: float):
        """
        Trains the neural network

        Parameters:
        - epochs: number of epochs
        - X_train: numpy array of features
        - y_train: numpy array of targets
        - learning_rate: float
        """
        input_layer = self.create_layer(X_train.shape[1])
        hidden_layer1 = self.create_layer(2)
        hidden_layer2 = self.create_layer(2)
        output_layer = self.create_layer(1)

        weights1 = self.create_fully_connected_weights(input_layer.size, hidden_layer1.size)
        weights2 = self.create_fully_connected_weights(hidden_layer1.size, hidden_layer2.size)
        weights3 = self.create_fully_connected_weights(hidden_layer2.size, output_layer.size)

        for _ in range(epochs):
            total_cost = 0
            # do forward and backward propagation for each input row
            for i in range(X_train.shape[0]):
                # forward
                input_layer = X_train[i].reshape(1, -1)
                hidden_layer1, hidden_layer_activated1 = self.forward(input_layer, weights1)
                hidden_layer2, hidden_layer_activated2 = self.forward(hidden_layer_activated1, weights2)
                output_layer, output_layer_activated = self.forward(hidden_layer_activated2, weights3)

                cost = (output_layer[0][0] - y_train[i]) ** 2
                total_cost += cost

                # backward
                dcdl_output = 2 * (output_layer - y_train[i].reshape(1, -1)) * 1
                dcdl_hidden2, dcdw3 = self.backward(dcdl_output, hidden_layer_activated2, hidden_layer2, weights3)
                dcdl_hidden1, dcdw2 = self.backward(dcdl_hidden2, hidden_layer_activated1, hidden_layer1, weights2)
                dcdl_input, dcdw1 = self.backward(dcdl_hidden1, input_layer, input_layer, weights1)

                weights1 -= learning_rate * dcdw1
                weights2 -= learning_rate * dcdw2
                weights3 -= learning_rate * dcdw3

            print(cost)
        self.weights = [weights1, weights2, weights3]

    def predict(self, x: np.ndarray):
        # do forward propagation
        layer = x
        output_layer = None
        for weights in self.weights:
            hidden_layer, hidden_layer_activated = self.forward(layer, weights)
            layer = hidden_layer_activated
            output_layer = hidden_layer

        return output_layer[0][0]
    
    def calculate_error(self, X: np.ndarray, y: np.array):
        """
        Parameters:

        - X: nd numpy array of features (m x n)
        - y: numpy array
        """

        err = 0
        for i in range(X.shape[0]):
            prediction = abs(self.predict(X[i].reshape(1, -1)) - y[i])
            actual = y[i]

            diff = abs(actual - prediction)
            err += diff**2

        return err / X.shape[0]

        

if __name__ == "__main__":
    nn = NN()

    X_train = np.array([[1, 2, 3, 4], [1.5, 2, 2.4, 3.5], [3.2, 0.7, 1.5, 2.1], [-0.5, 1.6, -1, 4]])
    y_train = np.array([2.25, 2.725, 2.675, 0.9])

    X_test = np.array([[2, 3, 4, 5], [-1, 2, 3, 4]])
    y_test = np.array([4, 1.75])

    nn.train(50, X_train, y_train, 0.001)
