# this script is used to generate images from mnist dataset and save them to a folder

import numpy as np
from PIL import Image
from keras.datasets import mnist
import os

_, (test_X, test_y) = mnist.load_data()

def save_images(X, y, folder):
    """
    Saves mnist images to a folder, This might be useful if someone wants to see the images and predictions

    Parameters:
    - X: numpy array of shape (n, 28, 28)
    - y: numpy array of shape (n,)
    """
    os.makedirs(folder, exist_ok=True)
    for i in range(len(X)):
        img = Image.fromarray(X[i].astype(np.uint8), 'L')
        img.save(f"{folder}/{i}_{y[i]}.png")

save_images(test_X[:10], test_y[:10], "mnist_png")