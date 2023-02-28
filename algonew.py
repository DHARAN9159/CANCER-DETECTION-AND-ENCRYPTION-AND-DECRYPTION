import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import matplotlib.pyplot as plt
import shutil
import os
import glob


# model


ROOT_DIR='data/brain_tumor_dataset'
number_of_images={}

for dir in os.listdir(ROOT_DIR):
  number_of_images [dir] = len(os.listdir(os.path.join(ROOT_DIR ,dir)))
print(number_of_images.items())
print(len(os.listdir('data/brain_tumor_dataset')))

if not os.path.exists("data/train"):
  os.mkdir("data/train")
  for dir in os.listdir(ROOT_DIR):
      os.makedirs("data/train/"+dir)
      for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR , dir)),size = (math.floor(70/100*number_of_images[dir])-5), replace=False):
        O = os.path.join(ROOT_DIR, dir, img)
        D = os.path.join("data/train", dir)
        shutil.copy(O,D)
        os.remove(O)

else:
    print("the folder exists")

def datafolder(p,split):
    if not os.path.exists("data/"+p):
        os.mkdir("data/"+p)
        for dir in os.listdir(ROOT_DIR):
            os.makedirs("data/"+p+"/"+dir)
            for img in np.random.choice(a=os.listdir(os.path.join(ROOT_DIR,dir)),
                                        size=(math.floor(split* number_of_images[dir])-5),replace=False):
                O = os.path.join(ROOT_DIR,dir,img)
                D = os.path.join("data/"+p,dir)
                shutil.copy(O,D)
                os.remove(O)

    else:
        print( f"{p} the folder exists")

datafolder("train",0.7)
datafolder("val",0.15)
datafolder("test",0.15)

from sklearn.model_selection import R
# model
# from tensorflow.keras.models import load_model
# import keras
# from keras.layers import Conv2D , MaxPool2D, Dropout,Flatten,Dense,BatchNormalization,GlobalAveragePooling2D
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from tensorflow.keras.utils import load_img
# from tensorflow.keras.utils import img_to_array
# from keras.models import load_model

#
# model = Sequential()
# model.add(Conv2D(16,(3 ,3),activation='relu',input_shape=(224,224,3)))
# model.add(Conv2D(36,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(MaxPool2D(pool_size=(2,2)))
# #
# model.add(Dropout(rate=0.25))
# model.add(Flatten())
# model.add(Dense(units=64,activation='relu'))
# model.add(Dropout(rate=0.25))
# model.add(Dense(units= 1,activation='sigmoid'))
# print(model.summary())
#
# model.compile(optimizer='adam', loss = 'keras.losses.binary_crossentropy', metrics=['accuracy'])
# #
# from keras.preprocessing.image import ImageDataGenerator
# from keras.applications import vgg16
# from keras.models import Model
# from keras.layers import Dense, MaxPool2D, Conv2D
# import keras
#
# train_path = "data/train"
# valid_path = "data/val"
# test_path = "data/val"

#
# def preprocessingimage1(path):
#     image_data = ImageDataGenerator( zoom_range= 0.2, horizontal_flip= True, shear_range= 0.2 , rescale= 1/255 )
#     image = image_data.flow_from_directory(directory= train_path , target_size=(224,224),batch_size=32,class_mode='binary')
#     return image
#
# train_data=preprocessingimage1(train_path)
#
# def prepocessingimage2(path):
#
#     image_data= ImageDataGenerator(rescale= 1/255 )
#     image = image_data.flow_from_directory(directory= valid_path , target_size=(224,224),batch_size=32,class_mode='binary')
#     return image
#
# test_data=prepocessingimage2(test_path)
# val_data=prepocessingimage2(valid_path)
# #
# # test_data_gen = ImageDataGenerator(preprocessing_function= vgg16.preprocess_input, rescale= 1/255 )
# # test = train_data_gen.flow_from_directory(directory= test_path , target_size=(224,224), shuffle= False)
#
#
# # print(train.class_indices)
# # t_img , label = train.next()
# # # while True:
# # def plotImages(img_arr, label):
# #
# #     for im, l in zip(img_arr,label) :
# #         plt.figure(figsize= (5,5))
# #         plt.imshow(im, cmap = 'rainbow_r')
# #         plt.title(im.shape)
# #         plt.axis = False
# #         plt.show()
# #
# #
# # print(plotImages(t_img, label))
# # #
#
#
#
# from keras.applications.vgg16 import VGG16
# from keras.layers import Flatten , Dense, Dropout , MaxPool2D
#
#
# vgg = VGG16( input_shape=(224,224,3), include_top= False)
# for layer in vgg.layers:
#   layer.trainable = False
#   x = Flatten()(vgg.output)
#   x = Dense(units=2, activation='sigmoid', name='predictions')(x)
#
#   model = Model(vgg.input, x)
#
#   # print(model.summary())
# #
# #
# # model.compile(optimizer='adam', loss = 'keras.losses.binary_crossentropy', metrics=['accuracy'])
#
# from keras.callbacks import ModelCheckpoint,EarlyStopping
# es=EarlyStopping(monitor="val_accuracy",min_delta=0.01,patience=3,verbose=1 ,mode="auto")
# mc=ModelCheckpoint(monitor="val_accuracy",filepath="modelcheck/bestmodel.h5",verbose=1,save_best_model=True,mode="auto")
# cd=[es,mc]
#
#
# hs = model.fit(train_data,steps_per_epoch=8,
# epochs=30,
#                          verbose=1,
#                          validation_data=val_data,
#                          validation_steps=16,
#                          callbacks=cd)
# hist = model.fit_generator(train_data, steps_per_epoch= 10, epochs= 30, validation_data= val_data, validation_steps= 16, callbacks=[es,mc])
# print(hist)
#
#