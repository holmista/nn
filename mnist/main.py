from nn import NN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("image_path", help="Path to the image to predict")
args = parser.parse_args()

nn = NN.load_model("./model")
print(nn.predict_from_image(args.image_path))
