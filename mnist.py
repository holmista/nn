from keras.datasets import mnist
import numpy as np
import mnistnn

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

nn = mnistnn.NN()

only_10_train_X = train_X[:10]
only_10_train_y = train_y[:10]

nn.train(10, only_10_train_X, only_10_train_y, 0.01)

# nn.train(1000, train_X, train_y, 0.01)

