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
import cv2
import math
import matplotlib.pyplot as plt

import os
import pandas as pd
import scipy.ndimage
from skimage import measure, morphology
from skimage.transform import rotate
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import tensorflow as tf

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.conf import settings


# train_path = "data/train"
# valid_path = "data/val"
# test_path = "data/val"
#
#
# train_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input , zoom_range= 0.2, horizontal_flip= True, shear_range= 0.2 , rescale= 1./255)
# train = train_data_gen.flow_from_directory(directory= train_path , target_size=(224,224))
#
#
# validation_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input , rescale= 1./255 )
# valid = validation_data_gen.flow_from_directory(directory= valid_path , target_size=(224,224))
#
# test_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input, rescale= 1./255 )
# test = train_data_gen.flow_from_directory(directory= test_path , target_size=(224,224), shuffle= False)
model = load_model("modelcheck/bestmodel.h5")
#
# acc = model.evaluate_generator(generator= test)[1]
#
# print(f"The accuracy of your model is = {acc} %")


def get_img_array(img_path):

    path = img_path
    img = image.load_img(path, target_size=(224, 224))
    x= image.img_to_array(img)
    x=x/255
    x=  np.expand_dims(x, axis=0)
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    global a
    a=int(preds)
    print(a)
    return a
path = "data/val/normal/kidney_normal_0019.jpg"
get_img_array(path)
# class_type = {1:'NO Cancer',  0: 'Cancer'}
if a == 0:
    print("cancer")
else:
    print("No cancer")

b = plt.imread(path)
plt.imshow(b, cmap="viridis")
plt.title("Original image")
plt.show()
print(plt.show())


# # res = np.argmax(model.predict(image123))
# print(f"The given  image is of type = {res}")
# print()
# print(f"The chances of image being  Cancer is : {model.predict(image123)[0][0]*100} percent")
# print()
# print(f"The chances of image being No Cancer is : {model.predict(image123)[0][1]*100} percent")
#
# # to display the image
# plt.imshow(image123[0], cmap = "gray")
# plt.title("input image")
# print(plt.show())