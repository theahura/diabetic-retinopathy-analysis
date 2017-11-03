import keras
from keras.models import load_model
from keras.utils import plot_model

import argparse
from PIL import Image
import numpy as np

from read_images import read_images



def main():
    parser = argparse.ArgumentParser(description="Predict Retinopathy from model")
    parser.add_argument("model_file", help="HDF5 file containing the trained model")
    parser.add_argument("image", help="Diabetic Retinopathy image to classify")
    args = parser.parse_args()

    model_file = args.model_file
    image_file = args.image

    model = load_model(model_file)

    image = Image.open(image_file)


    prediction = model.predict(image_data[0:2])
    print prediction

    # plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

if __name__ == "__main__":
    main()