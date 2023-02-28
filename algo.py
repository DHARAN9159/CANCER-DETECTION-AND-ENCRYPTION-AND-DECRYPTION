    import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import matplotlib.pyplot as plt
import shutil
import os

import glob


# # model


# ROOT_DIR='data/cancerdata'
# number_of_images={}
#
# for dir in os.listdir(ROOT_DIR):
#   number_of_images [dir] = len(os.listdir(os.path.join(ROOT_DIR ,dir)))
# print(number_of_images.items())
# print(len(os.listdir('data/cancerdata')))

# if not os.path.exists("data/train"):
#   os.mkdir("data/train")
#   for dir in os.listdir(ROOT_DIR):
#       os.makedirs("data/train/"+dir)
#       for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR , dir)),size = (math.floor(70/100*number_of_images[dir])-5), replace=False):
#         O = os.path.join(ROOT_DIR, dir, img)
#         D = os.path.join("data/train", dir)
#         shutil.copy(O,D)
#         os.remove(O)
#
# else:
#     print("the folder exists")
#
# def datafolder(p,split):
#     if not os.path.exists("data/"+p):
#         os.mkdir("data/"+p)
#         for dir in os.listdir(ROOT_DIR):
#             os.makedirs("data/"+p+"/"+dir)
#             for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR,dir)),
#                                         size=(math.floor(split* number_of_images[dir])-5),replace=False):
#                 O = os.path.join(ROOT_DIR,dir,img)
#                 D = os.path.join("data/"+p,dir)
#                 shutil.copy(O,D)
#                 os.remove(O)
#
#     else:
#         print( f"{p} the folder exists")
#
# datafolder("train",0.7)
# datafolder("val",0.15)
# datafolder("test",0.15)

train_path = "data/train"
valid_path = "data/val"
test_path = "data/val"

from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.models import Model
from keras.layers import Dense, MaxPool2D, Conv2D
import keras
import seaborn as sns
import pandas as pd
import tensorflow as tf



train_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input , zoom_range= 0.2, horizontal_flip= True, shear_range= 0.2 , rescale= 1./255)
train = train_data_gen.flow_from_directory(directory= train_path , target_size=(224,224))


validation_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input , rescale= 1./255 )
valid = validation_data_gen.flow_from_directory(directory= valid_path , target_size=(224,224))

test_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input, rescale= 1./255 )
test = train_data_gen.flow_from_directory(directory= test_path , target_size=(224,224), shuffle= False)


print(train.class_indices)
class_type = {0:'NO Cancer',  1: 'Cancer'}

t_img, label = train.next()


# def plotImages(img_arr, label):
#     for im, l in zip(img_arr, label):
#         plt.figure(figsize=(5, 5))
#         plt.imshow(im, cmap='gray')
#         plt.title(im.shape)
#         plt.axis = False
#         # plt.show()


# plotImages(t_img, label)
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten , Dense, Dropout , MaxPool2D
vgg = VGG16( input_shape=(224,224,3), include_top= False)

for layer in vgg.layers:
  layer.trainable = False
  x = Flatten()(vgg.output)
  x = Dense(units=2, activation='sigmoid', name='predictions')(x)

  model = Model(vgg.input, x)

  # print(model.summary())

model.compile(optimizer='adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

es = EarlyStopping(monitor= "val_accuracy" , min_delta= 0.01, patience= 3, verbose=1)
mc = ModelCheckpoint(filepath="modelcheck/bestmodel.h5", monitor="val_accuracy", verbose=1, save_best_only= True)


hist = model.fit_generator(train, steps_per_epoch= 8, epochs= 10, validation_data= valid , validation_steps= 32, callbacks=[es,mc])

from keras.models import load_model
model = load_model("modelcheck/bestmodel.h5")
h = hist.history
h.keys()
plt.plot(h['accuracy'])
plt.plot(h['val_accuracy'] , c = "red")
plt.title("acc vs v-acc")
plt.show()

#
plt.plot(h['loss'])
plt.plot(h['val_loss'] , c = "red")
plt.title("loss vs v-loss")
plt.show()

import matplotlib.pyplot as plt

# accuracy = [0.8, 0.9, 0.85, 0.93, 0.87]
# epochs = [1, 2, 3, 4, 5]
# epochs=[1,2,3,4,5,6,7,8,9,10]
# plt.plot(epochs ,h['accuracy'], 'b', label='Accuracy')
# plt.title('Accuracy over epochs')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()


# plt.plot(h['loss'] ,h['accuracy'], 'b', label='Accuracy')
# plt.title('Accuracy over epochs')
# plt.xlabel('loss')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()
# plt.plot(h['accuracy'] ,h['val_accuracy'], 'b', label='Accuracy')
# plt.title('Accuracy over epochs')
# plt.xlabel('accuracy')
# plt.ylabel('val_accuracy')
# plt.legend()
# plt.show()

#
sns.distplot(h['accuracy'] , hist=False, color="blue")
sns.distplot(h['val_accuracy'], hist=False, color="orange")
# plt.show()




acc = model.evaluate_generator(generator= test)[1]

print(f"The accuracy of your model is = {acc} %")

import keras.utils as image


def get_img_array(img_path):
    """
    Input : Takes in image path as input
    Output : Gives out Pre-Processed image
    """
    path = img_path
    img = image.load_img(path, target_size=(224, 224, 3))
    img = image.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)

    return img


path = "data/test/cancer/kidney_tumor_0009.jpg"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
image123 = get_img_array(path)
print(image123)

res = class_type[np.argmax(model.predict(image123))]
print(f"The given  image is of type = {res}")
print()
print(f"The chances of image being  Cancer is : {model.predict(image123)[0][0]*100} percent")
print()
print(f"The chances of image being No Cancer is : {model.predict(image123)[0][1]*100} percent")

# to display the image
plt.imshow(image123[0], cmap = "gray")
plt.title("input image")
print(plt.show())


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]


    grads = tape.gradient(class_channel, last_conv_layer_output)


    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))


    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()




import matplotlib.cm as cm

from IPython.display import Image, display


def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    """
    img input shoud not be expanded
    """

    # Load the original image


    img = tf.keras.preprocessing.image.load_img(img_path)
    img = tf.keras.preprocessing.image.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    display(Image(cam_path))

def image_prediction_and_visualization(path, last_conv_layer_name="block5_conv3", model=model):
    """
    input:  is the image path, name of last convolution layer , model name
    output : returs the predictions and the area that is effected
    """

    img_array = get_img_array(path)

    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

    img = get_img_array(path)

    res = class_type[np.argmax(model.predict(img))]
    print(f"The given X-Ray image is of type = {res}")
    print()
    print(f"The chances of image being normal is : {model.predict(img)[0][0] * 100} %")
    print(f"The chances of image being cancer is : {model.predict(img)[0][1] * 100} %")

    print()
    print("image ")

        # function call
    save_and_display_gradcam(path, heatmap)

    print()
    print("the original input image")
    print()

    a = plt.imread(path)
    plt.imshow(a, cmap="gray")
    plt.title("Original image")
    plt.show()
    print(plt.show())

path ="data/test/cancer/kidney_tumor_0009.jpg"

image_prediction_and_visualization(path)

