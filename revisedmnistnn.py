import numpy as np

class NN:
    def __init__(self):
        self.weights = []

    def create_layer(self, n: int):
        """
        Creates and returns a zero numpy array of shape 1 x n

        Parameters:
        - n amount of nodes in a layer
        """
        layer = np.zeros(n).reshape(1, -1)
        return layer
    
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

    def forward(self, a: np.ndarray, w: np.ndarray, b: np.ndarray):
        """
        Computes forward pass for a fully connected layer for a minibatch

        Parameters:
        - a: A minibatch of Inputs(n x m), activations of previous layer
        - w: A numpy ndarray(m x d), weights connecting this layer to the next layer
        - b: A numpy ndarray(1 x d), biases for this layer

        Returns a tuple of:
        - z: nd numpy array(n x m)
        - activated: the same as the but activated with relu function
        """
        z = a @ w + b
        activated = self.relu_activation(z)
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
        
        input_layer = self.create_layer(X_train.shape[1])
        hidden_layer1 = self.create_layer(16)
        hidden_layer2 = self.create_layer(16)
        output_layer = self.create_layer(3)

        weights1 = self.create_fully_connected_weights(input_layer.size, hidden_layer1.size)
        weights2 = self.create_fully_connected_weights(hidden_layer1.size, hidden_layer2.size)
        weights3 = self.create_fully_connected_weights(hidden_layer2.size, output_layer.size)

        # create biases
        bias1 = np.zeros(hidden_layer1.size).reshape(1, -1)
        bias2 = np.zeros(hidden_layer2.size).reshape(1, -1)
        bias3 = np.zeros(output_layer.size).reshape(1, -1)

        batches = self.create_batches(X_train, y_train, batch_size)

        for epoch in range(epochs):
            total_cost = 0
            for X_batch, y_batch in batches:
                # forward
                input_layer = X_batch
                z1, hidden_layer_activated1 = self.forward(input_layer, weights1, bias1)
                z2, hidden_layer_activated2 = self.forward(hidden_layer_activated1, weights2, bias2)
                z3, output_layer_activated = self.forward(hidden_layer_activated2, weights3, bias3)
                output_layer_activated = self.softmax_activation(z3)

                # cost
                cost = self.cross_entropy_cost(y_batch, output_layer_activated, weights3, regularization_coefficient)

                # backward
                dprobs = self.softmax_gradient(y_batch, output_layer_activated)
                dhl2, dw3, db3 = self.backward(dprobs, hidden_layer_activated2, z2, weights3)
                dhl1, dw2, db2 = self.backward(dhl2, hidden_layer_activated1, z1, weights2)
                _, dw1, db1 = self.backward(dhl1, input_layer, input_layer, weights1)

                # regularization
                dw3 += regularization_coefficient * weights3
                dw2 += regularization_coefficient * weights2
                dw1 += regularization_coefficient * weights1

                # update weights
                weights1 -= learning_rate * dw1
                weights2 -= learning_rate * dw2
                weights3 -= learning_rate * dw3

                # update biases
                bias1 -= learning_rate * db1
                bias2 -= learning_rate * db2
                bias3 -= learning_rate * db3

                total_cost += cost
            if (epoch+1) % 1000 == 0:
                print(f"Epoch {epoch + 1}, cost: {total_cost}")

        self.weights = np.array([weights1, weights2, weights3])

if __name__ == "__main__":

    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    one_hot_encoded = np.zeros((y.size, K))
    one_hot_encoded[np.arange(y.size), y] = 1
    y = one_hot_encoded

    nn = NN()
    nn.train(90000, X, y, 1e-2, 100, 1e-4)






    


