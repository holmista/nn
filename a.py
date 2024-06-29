from keras.datasets import mnist

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors 
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# one hot encode train labels
for i in range(train_y.shape[0]):
    temp = [0]*10
    temp[train_y[i]] = 1
    train_y[i] = temp

# divide all images by 255 so that all pixel values are between 0 and 1
train_X = train_X / 255




print(train_y[0])