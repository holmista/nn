# import numpy as np

# N = 100 # number of points per class
# D = 2 # dimensionality
# K = 3 # number of classes
# X = np.zeros((N*K,D)) # data matrix (each row = single example)
# y = np.zeros(N*K, dtype='uint8') # class labels
# for j in range(K):
#     ix = range(N*j,N*(j+1))
#     r = np.linspace(0.0,1,N) # radius
#     t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
#     X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
#     y[ix] = j

# # initialize parameters randomly
# h = 100 # size of hidden layer
# W = 0.01 * np.random.randn(D,h)
# b = np.zeros((1,h))
# W2 = 0.01 * np.random.randn(h,K)
# b2 = np.zeros((1,K))

# # some hyperparameters
# step_size = 1e-0
# reg = 1e-3 # regularization strength

# # gradient descent loop
# num_examples = X.shape[0]
# for i in range(10000):

# # evaluate class scores, [N x K]
#     hidden_layer = np.maximum(0, np.dot(X, W) + b) # note, ReLU activation
#     scores = np.dot(hidden_layer, W2) + b2

#     # compute the class probabilities
#     exp_scores = np.exp(scores)
#     probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]

#     # compute the loss: average cross-entropy loss and regularization
#     correct_logprobs = -np.log(probs[range(num_examples),y])
#     data_loss = np.sum(correct_logprobs)/num_examples
#     reg_loss = 0.5*reg*np.sum(W*W) + 0.5*reg*np.sum(W2*W2)
#     loss = data_loss + reg_loss
#     if i % 1000 == 0:
#         print ("iteration %d: loss %f" % (i, loss))

#     # compute the gradient on scores
#     dscores = probs
#     dscores[range(num_examples),y] -= 1
#     dscores /= num_examples

#     # backpropate the gradient to the parameters
#     # first backprop into parameters W2 and b2
#     dW2 = np.dot(hidden_layer.T, dscores)
#     db2 = np.sum(dscores, axis=0, keepdims=True)
#     # next backprop into hidden layer
#     dhidden = np.dot(dscores, W2.T)
#     # backprop the ReLU non-linearity
#     dhidden[hidden_layer <= 0] = 0
#     # finally into W,b
#     dW = np.dot(X.T, dhidden)
#     db = np.sum(dhidden, axis=0, keepdims=True)

#     # add regularization gradient contribution
#     dW2 += reg * W2
#     dW += reg * W

#     # perform a parameter update
#     W += -step_size * dW
#     b += -step_size * db
#     W2 += -step_size * dW2
#     b2 += -step_size * db2

# hidden_layer = np.maximum(0, np.dot(X, W) + b)
# scores = np.dot(hidden_layer, W2) + b2
# predicted_class = np.argmax(scores, axis=1)
# print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

import numpy as np

class NN:
    def __init__(self):
        self.weights = None

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
        # rd = self.relu_gradient(unactivated)
        # dcdl = (dprev @ w.T) * rd
        # dcdw = activated.T @ dprev
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
        normalized_z = z - z.max()
        exps = np.exp(normalized_z)
        activated_z = exps / exps.sum(axis=1, keepdims=True)
        return activated_z
    
    def cross_entropy_cost_and_gradient(self, y_true: np.ndarray, z: np.ndarray, w: np.ndarray, reg: float):
        """
        Calculates cross-entropy cost of a softmax layer and also calculates gradient of softmax layer and weights connected to it

        Parameters:
        - y_true: A numpy nd array(n x m) of one-hot encoded labels. This is a minibatch of size n, m is the amount of classes
        - z: A numpy nd array(n x d), activation of the previous layer. This is a minibatch of size n, d is the amount of neurons
        - w: A numpy nd array(d x m). This are the weights connecting previous layer to softmax later
        - reg: Regularization strength

        Returns:
        - dprobs: The gradient of the softmax layer
        - dw: The gradient of the weights
        - cost: The cross-entropy cost
        """
        # calculate probabilities via applying softmax
        shifted_z = z - np.max(z, axis=1, keepdims=True)
        exps = np.exp(shifted_z)
        probs = exps / (np.sum(exps, axis=1, keepdims=True) + 1e-8)

        # calculate cost
        n = y_true.shape[0]
        cost = -np.sum(y_true * np.log(probs + 1e-15))
        # regularization
        cost = cost / n + 0.5 * reg * np.sum(w**2)

        # calculate gradients
        dprobs = probs.copy()
        dprobs -= y_true
        dprobs /= n

        # dw = z.T @ dprobs
        # dw += reg * w
        return dprobs, cost
    
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
        output_layer = self.create_layer(10)

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

                # cost
                dprobs, cost = self.cross_entropy_cost_and_gradient(y_batch, z3, weights3, regularization_coefficient)
                dhl2, dw3, db3 = self.backward(dprobs, hidden_layer_activated2, z2, weights3)
                dhl1, dw2, db2 = self.backward(dhl2, hidden_layer_activated1, z1, weights2)
                _, dw1, db1 = self.backward(dhl1, input_layer, input_layer, weights1)

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
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch + 1}, cost: {total_cost}")
            # print(f"Epoch {epoch + 1}, cost: {total_cost}")

        self.weights = [weights1, weights2, weights3]

    def predict(self, X: np.ndarray):
        """
        Predicts the labels of the input data

        Parameters:
        - X: A numpy nd array(n x m) of inputs

        Returns:
        - A numpy nd array(n x 1) of labels
        """
        weights1, weights2, weights3 = self.weights
        bias1 = np.zeros(weights1.shape[1]).reshape(1, -1)
        bias2 = np.zeros(weights2.shape[1]).reshape(1, -1)
        bias3 = np.zeros(weights3.shape[1]).reshape(1, -1)

        z1, hidden_layer_activated1 = self.forward(X, weights1, bias1)
        z2, hidden_layer_activated2 = self.forward(hidden_layer_activated1, weights2, bias2)
        z3, output_layer_activated = self.forward(hidden_layer_activated2, weights3, bias3)

        return np.argmax(output_layer_activated, axis=1)
    
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

    nn.train(200, train_X, train_y, 1e-3, 150, 1e-3)
    y_pred = nn.predict(test_X)
    accuracy = nn.calculate_accuracy(np.argmax(test_y, axis=1), y_pred)
    print(f"Accuracy: {accuracy}")
    # N = 100 # number of points per class
    # D = 2 # dimensionality
    # K = 3 # number of classes
    # X = np.zeros((N*K,D)) # data matrix (each row = single example)
    # y = np.zeros(N*K, dtype='uint8') # class labels
    # for j in range(K):
    #     ix = range(N*j,N*(j+1))
    #     r = np.linspace(0.0,1,N) # radius
    #     t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    #     X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    #     y[ix] = j

    # one_hot_encoded = np.zeros((y.size, K))
    # one_hot_encoded[np.arange(y.size), y] = 1
    # y = one_hot_encoded

    # nn = NN()
    # nn.train(90000, X, y, 1e-2, 100, 1e-4)







    


