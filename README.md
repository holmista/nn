# MNIST Neural Network Experiment

This project is an experimental implementation of a neural network for MNIST digit classification. It's currently a work in progress and serves as a template or starting point for further development.

## Project Description

This neural network is designed to classify handwritten digits from the MNIST dataset. The current implementation includes:

- Loading and preprocessing the MNIST dataset
- One-hot encoding of labels
- Normalization of pixel values
- Reshaping of images to 1D arrays
- A basic neural network implementation in the `mnistnn` module

This implementation predicts testing data with up to 90% accuracy.

## Installation

1. Install pyenv for managing virtual environments:
```pip install pyenv```

2. Create a virtual environment:
```pyenv virtualenv mnistnn```

3. Activate the virtual environment:
```pyenv activate mnistnn```

4. Install the required packages:
```pip install -r requirements.txt```

## Training the model

To train the model, run the following command:
```python3 mnist/nn.py```
There's already initialized network architecture but you can change it for experimentation.
Running the above file will also save the model to the `model` directory. The model can later be loaded and used for prediction.

## Predicting with the model

To predict with the model, run the following command:
```python3 mnist/main.py```
This will load the model from the `model` directory and predict the labels for some of the testing data.
