import numpy as np
import os
from PIL import Image

class NN:
    def __init__(self):
        self.weights = []
        self.layers = []
        self.biases = []
        self.activations = []

    def create_layer(self, n: int):
        """
        Creates and returns a zero numpy array of shape 1 x n

        Parameters:
        - n amount of nodes in a layer
        """
        layer = np.zeros(n).reshape(1, -1)
        self.layers.append(layer)
        return layer
    
    def create_bias(self, n: int):
        """
        Creates and returns a zero numpy array of shape 1 x n

        Parameters:
        - n amount of nodes in a layer
        """
        bias = np.zeros(n).reshape(1, -1)
        self.biases.append(bias)
        return bias
    
    def define_activations(self, activations: list):
        """
        Defines the activation functions for each layer
        The activations are defined in the order of the layers

        Parameters:
        - activations: A list of activation functions
        """
        self.activations = activations
    
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
        self.weights.append(weights)
        return weights
    
    def create_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int):
        """
        Breaks data into batches

        Parameters:
        - X: A numpy nd array(n x m) of inputs
        - y: A numpy nd array(n x 1) of labels
        - batch_size: The size of the batches

        Returns:
        - A list of tuples of (X_batch, y_batch)
        """
        n = X.shape[0]
        batches = []
        for i in range(0, n, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            batches.append((X_batch, y_batch))
        return batches
    
    def _resolve_activation(self, activation: str):
        if activation == "relu":
            return self.relu_activation
        elif activation == "softmax":
            return self.softmax_activation
        else:
            raise ValueError("Activation not supported")

    def forward(self, a: np.ndarray, w: np.ndarray, b: np.ndarray, activation: str):
        """
        Computes forward pass for a fully connected layer for a minibatch

        Parameters:
        - a: A minibatch of Inputs(n x m), activations of previous layer
        - w: A numpy ndarray(m x d), weights connecting this layer to the next layer
        - b: A numpy ndarray(1 x d), biases for this layer
        - activation: The activation function to be used

        Returns a tuple of:
        - z: nd numpy array(n x m)
        - activated: the same as the but activated with relu function
        """
        z = a @ w + b
        activation_function = self._resolve_activation(activation)
        activated = activation_function(z)
        return z, activated
    
    def backward(self, dprev: np.ndarray, activated: np.ndarray, unactivated: np.ndarray, w: np.ndarray):
        """
        Computes backward pass for a fully connected layer for a minibatch

        Parameters:
        - dprev: A numpy nd array of previous gradient(n x m)
        - activated: A numpy nd array(n x d) representing a relu activated layer
        - unactivated: the same as activated except activation is not applied
        - w: A numpy nd array of weights(d x m) connecting this layer to the previous layer

        Returns:
        - dcdl: gradient with respect to layer
        - dcdw: gradient with respect to w
        """
        rd = self.relu_gradient(unactivated)
        dcdl = dprev.dot(w.T) * rd
        dcdw = activated.T.dot(dprev)
        dcdb = np.sum(dprev, axis=0, keepdims=True)

        return dcdl, dcdw, dcdb
    
    def relu_activation(self, z: np.ndarray):
        """
        Applies relu activation on an input

        Parameters:
        - x: A numpy nd array(n x m). This is a minibatch of size n

        Returns:
        - activated_z: A numpy nd array(n x m)
        """
        activated_z = np.maximum(0, z)
        return activated_z
    
    def relu_gradient(self, z: np.ndarray):
        """
        Computes the gradient of the relu activation function.

        Parameters:
        - z: A numpy nd array(n x m). This is a minibatch of size n

        Returns:
        - gradient: The derivative of relu applied to z.
        """
        return (z > 0).astype(float)
    
    def softmax_activation(self, z: np.ndarray):
        """
        Applies softmax activation on an input

        Parameters:
        - z: A numpy nd array(n x m). This is a minibatch of size n, m is the amount of classes

        Returns:
        - activated_z: A numpy nd array(n x m)
        """
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shifted_z)
        probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-8)
        return probs
    
    def cross_entropy_cost(self, y_true: np.ndarray, a: np.ndarray, w: np.ndarray, reg: float):
        """
        Calculates cross-entropy cost of a softmax layer

        Parameters:
        - y_true: A numpy nd array(n x m) of one-hot encoded labels. This is a minibatch of size n, m is the amount of classes
        - a: A numpy nd array(n x m), activation of the softmax layer. This is a minibatch of size n, m is the amount of classes
        - w: A numpy nd array(d x m). This are the weights connecting previous layer to softmax later
        - reg: Regularization strength

        Returns:
        - cost: The cross-entropy cost
        """
        n = y_true.shape[0]
        cost = -np.sum(y_true * np.log(a + 1e-15))
        cost = cost / n + 0.5 * reg * np.sum(w**2)
        return cost
    
    def softmax_gradient(self, y_true: np.ndarray, a: np.ndarray):
        """
        Calculates the gradient of the softmax layer, assuming cross-entropy cost was used

        Parameters:
        - y_true: A numpy nd array(n x m) of one-hot encoded labels. This is a minibatch of size n, m is the amount of classes
        - a: A numpy nd array(n x d), activation of the previous layer. This is a minibatch of size n, d is the amount of neurons

        Returns:
        - da: The gradient of the softmax layer
        """
        n = y_true.shape[0]
        da = a.copy()
        da -= y_true
        da /= n
        return da
    
    def train(
            self, 
            epochs: int, 
            X_train: np.ndarray, 
            y_train: np.ndarray, 
            learning_rate: float, 
            batch_size: int, 
            regularization_coefficient:int):
        
        batches = self.create_batches(X_train, y_train, batch_size)

        for epoch in range(epochs):
            total_cost = 0
            for X_batch, y_batch in batches:
                # forward
                current_layer = X_batch
                activated_layers = [X_batch]
                for i in range(len(self.layers) - 1):
                    z, activated = self.forward(current_layer, self.weights[i], self.biases[i], self.activations[i])
                    self.layers[i + 1] = z
                    activated_layers.append(activated)
                    current_layer = activated

                # cost
                cost = self.cross_entropy_cost(y_batch, activated_layers[-1], self.weights[-1], regularization_coefficient)
                dlast_layer = self.softmax_gradient(y_batch, activated_layers[-1])

                # remove last layer from activated layers since it is the output layer and we don't need it
                activated_layers = activated_layers[:-1]

                # backward
                weights_gradients = []
                biases_gradients = []
                for i in range(len(self.layers) - 2, -1, -1):
                    dhln, dwn, dbn = self.backward(dlast_layer, activated_layers[i], self.layers[i], self.weights[i])
                    dlast_layer = dhln
                    weights_gradients.append(dwn)
                    biases_gradients.append(dbn)

                weights_gradients = weights_gradients[::-1]
                biases_gradients = biases_gradients[::-1]

                # regularization
                for i in range(len(self.weights)):
                    weights_gradients[i] += regularization_coefficient * self.weights[i]

                # update weights
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * weights_gradients[i]

                # update biases
                for i in range(len(self.biases)):
                    self.biases[i] -= learning_rate * biases_gradients[i]

                total_cost += cost
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch + 1}, cost: {total_cost}")

    def predict(self, X: np.ndarray):
        """
        Predicts the labels of the input data

        Parameters:
        - X: A numpy nd array(n x m) of inputs

        Returns:
        - A numpy nd array(n x 1) of labels
        """
        current_layer = X
        for i in range(len(self.layers) - 1):
            z, activated = self.forward(current_layer, self.weights[i], self.biases[i], self.activations[i])
            current_layer = activated

        return np.argmax(current_layer, axis=1)
    
    def predict_from_image(self, path: str):
        """
        Predicts the label of an image. This assumes that the image is a 28 x 28 grayscale image

        Parameters:
        - path: The path to the image
        """
        img = Image.open(path).convert("L")
        img_array = np.array(img)
        img_array = img_array / 255
        img_array = img_array.reshape(1, -1)
        return self.predict(img_array)[0]

    
    def save_model(self, path: str):
        """
        Saves model weights, biases and activations inside the given folder

        Parameters:
        - path: The path to the folder where the model will be saved
        """
        os.makedirs(f"{path}/weights", exist_ok=True)
        os.makedirs(f"{path}/biases", exist_ok=True)
        os.makedirs(f"{path}/architecture", exist_ok=True)
        for i in range(len(self.weights)):
            self.weights[i].tofile(f"{path}/weights/weights_{i}.csv", sep = ',')
            self.biases[i].tofile(f"{path}/biases/biases_{i}.csv", sep = ',')

        activations = np.array(self.activations)
        activations.tofile(f"{path}/activations.csv", sep = ',', format = '%s')

        layers_architecture = np.array([layer.shape[1] for layer in self.layers])
        layers_architecture.tofile(f"{path}/architecture/layers_architecture.csv", sep = ',')

        weights_architecture = np.array([[weight.shape[0], weight.shape[1]] for weight in self.weights])
        weights_architecture.tofile(f"{path}/architecture/weights_architecture.csv", sep = ',')

        biases_architecture = np.array([bias.shape[1] for bias in self.biases])
        biases_architecture.tofile(f"{path}/architecture/biases_architecture.csv", sep = ',')

    @staticmethod
    def load_model(path: str):
        """
        Creates an instance of the model from the weights, biases and activations saved in the given folder

        Parameters:
        - path: The path to the folder where the model is saved

        Returns:
        - An instance of the model
        """
        nn = NN()
        weights = []
        biases = []
        layers_architecture = np.fromfile(f"{path}/architecture/layers_architecture.csv", sep = ',').astype(int)
        weights_architecture = np.fromfile(f"{path}/architecture/weights_architecture.csv", sep = ',').reshape(-1, 2).astype(int)
        biases_architecture = np.fromfile(f"{path}/architecture/biases_architecture.csv", sep = ',').astype(int)

        for i in layers_architecture:
            nn.create_layer(i)

        for i in range(weights_architecture.shape[0]):
            weights.append(np.fromfile(f"{path}/weights/weights_{i}.csv", sep=',').reshape(weights_architecture[i][0], weights_architecture[i][1]))
            biases.append(np.fromfile(f"{path}/biases/biases_{i}.csv", sep=','))

        activations = np.genfromtxt(f"{path}/activations.csv", dtype="str", delimiter=',')

        nn.weights = weights
        nn.biases = biases
        nn.activations = activations
         
        return nn
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Calculates the accuracy of the model in percentages

        Parameters:
        - y_true: A numpy nd array(n x 1) of true labels
        - y_pred: A numpy nd array(n x 1) of predicted labels

        Returns:
        - The accuracy of the model
        """
        return np.mean(y_true == y_pred) * 100

if __name__ == "__main__":
    from keras.datasets import mnist
    import numpy as np

    # loading the dataset. p.s. too much pain to use raw version, using this already premade instead
    (train_X, train_y), (test_X, test_y) = mnist.load_data()

    # one-hot encode train labels
    train_y_encoded = np.zeros((train_y.shape[0], 10))
    for i in range(train_y.shape[0]):
        train_y_encoded[i, train_y[i]] = 1

    train_y = train_y_encoded

    # One-hot encode test labels
    test_y_encoded = np.zeros((test_y.shape[0], 10))
    for i in range(test_y.shape[0]):
        test_y_encoded[i, test_y[i]] = 1

    test_y = test_y_encoded

    # divide all images by 255 so that all pixel values are between 0 and 1
    train_X = train_X / 255
    test_X = test_X / 255

    # reshape the images to be 1D
    train_X = train_X.reshape(train_X.shape[0], -1)
    test_X = test_X.reshape(test_X.shape[0], -1)

    nn = NN()

    input_layer = nn.create_layer(train_X.shape[1])
    hidden_layer1 = nn.create_layer(16)
    hidden_layer2 = nn.create_layer(16)
    output_layer = nn.create_layer(10)

    nn.activations = ["relu", "relu", "softmax"]

    weights1 = nn.create_fully_connected_weights(input_layer.size, hidden_layer1.size)
    weights2 = nn.create_fully_connected_weights(hidden_layer1.size, hidden_layer2.size)
    weights3 = nn.create_fully_connected_weights(hidden_layer2.size, output_layer.size)

    # create biases
    bias1 = nn.create_bias(hidden_layer1.size)
    bias2 = nn.create_bias(hidden_layer2.size)
    bias3 = nn.create_bias(output_layer.size)

    nn.train(200, train_X, train_y, 1e-3, 150, 1e-3)
    y_pred = nn.predict(test_X)
    accuracy = nn.calculate_accuracy(np.argmax(test_y, axis=1), y_pred)
    print(f"Accuracy: {accuracy}")
    nn.save_model("./model")