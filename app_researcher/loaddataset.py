import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, MaxPool2D, Conv2D
import keras
import seaborn as sns
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import keras.utils as image
import matplotlib.cm as cm
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def imagemodel():
    train_path = "data/train"
    valid_path = "data/val"
    test_path = "data/val"

    train_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, zoom_range=0.2,
                                        horizontal_flip=True, shear_range=0.2, rescale=1. / 255)
    train = train_data_gen.flow_from_directory(directory=train_path, target_size=(224, 224))

    validation_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, rescale=1. / 255)
    valid = validation_data_gen.flow_from_directory(directory=valid_path, target_size=(224, 224))

    test_data_gen = ImageDataGenerator(preprocessing_function=vgg16.preprocess_input, rescale=1. / 255)
    test = train_data_gen.flow_from_directory(directory=test_path, target_size=(224, 224), shuffle=False)

    # print(train.class_indices)
    # class_type = {0: 'NO Cancer', 1: 'Cancer'}

    t_img, label = train.next()
    from keras.layers import Flatten, Dense, Dropout, MaxPool2D
    from keras.applications.vgg16 import VGG16
    vgg = VGG16(input_shape=(224, 224, 3), include_top=False)

    for layer in vgg.layers:
        layer.trainable = False
        x = Flatten()(vgg.output)
        x = Dense(units=2, activation='sigmoid', name='predictions')(x)

        model = Model(vgg.input, x)

        # print(model.summary())

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    from keras.callbacks import EarlyStopping
    from keras.callbacks import ModelCheckpoint

    es = EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, verbose=1)
    mc = ModelCheckpoint(filepath="modelcheck/bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only=True)

    hist = model.fit_generator(train, steps_per_epoch=8, epochs=30, validation_data=valid, validation_steps=32,
                               callbacks=[es, mc])
