from nn import NN

nn = NN.load_model("./model")
print(nn.predict_from_image("mnist_png/3_0.png"))
