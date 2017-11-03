import argparse
import numpy as np

import keras

from util import dr_model, read_images, get_labels



#Design model

def main():
    parser = argparse.ArgumentParser(description="Load Diabetic Retinopathy images")
    parser.add_argument("train_data", help="folder containing training set")
    parser.add_argument("test_data", help="folder containing testing set")
    parser.add_argument("labels", help="csv file containing image labels")
    parser.add_argument("--model_output", metavar="o", help="path for model output")
    args = parser.parse_args()

    train_data_dir = args.train_data
    test_data_dir = args.test_data
    labels_file = args.labels
    model_output = args.model_output

    if model_output == None:
        model_output = 'output.hdf5'

    
    #Get training and test images
    train_images = read_images(train_data_dir)
    test_images = read_images(test_data_dir)


    #Extract data and labels for images in each set
    train_data = np.array([np.asarray(image, dtype=np.float32) for image in train_images.values()])
    train_labels = keras.utils.to_categorical(get_labels(train_images, labels_file), num_classes=5)

    test_data = np.array([np.asarray(image, dtype=np.float32) for image in test_images.values()])
    test_labels = keras.utils.to_categorical(get_labels(test_images, labels_file), num_classes=5)

    #Load the retinopathy model
    model = dr_model()

    #Fit the data to the model
    print "Training model"
    try:
        model.fit(train_data, train_labels, epochs=50, batch_size=32)
    except KeyboardInterrupt:
        print "Quitting and saving in {}".format(model_output)


    #Save model
    print "Saving in {}".format(model_output)
    model.save(model_output)

    #Evaluate the model
    print "Evaluating model"
    score = model.evaluate(test_data, test_labels, batch_size=16)

    print score


if __name__ == "__main__":
    main()

