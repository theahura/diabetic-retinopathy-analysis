import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, Flatten

import os
from PIL import Image
import pandas


def dr_model():
    """
        This is a neural network model for identifying diabetic retinopathy in 
        fundus photographs
    """

    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2), padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(5, activation='softmax'))


    model.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



def read_images(path):
    """
        Read images from folder 'path'
    """
    
    files = [os.path.join(path,f) for f in os.listdir(path)]
    images = dict([(f, Image.open(f)) for f in files])

    return images


def get_labels(image_names, label_csv):
    """
        Get retimopathy classification from csv
    """
    image_basenames = [os.path.splitext(os.path.basename(f))[0] for f in image_names]
    data = pandas.read_csv(label_csv)

    labels = []
    for name in image_basenames:
        level = data[data.image == name].level[0]
        labels.append(level)

        